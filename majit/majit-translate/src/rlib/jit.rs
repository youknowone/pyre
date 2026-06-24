//! Rust port of `rpython/rlib/jit.py:875-1024` — annotator / rtyper
//! support for the `jit_merge_point` / `can_enter_jit` /
//! `loop_header` markers.
//!
//! The runtime `JitDriver` class itself (rlib/jit.py:610-693) lives
//! on the pyre side as `PyPyJitDriver` (`pyre/pyre-jit/src/eval.rs`).
//! This module owns only the translator-side mirror metadata
//! (`JitDriverMeta`) plus the kwarg-validation / cache-union helper
//! consumed by `ExtRegistryEntry::EnterLeaveMarker` /
//! `ExtRegistryEntry::LoopHeader` (in
//! `translator/rtyper/extregistry.rs`).
//!
//! `ExtRegistryEntry::EnterLeaveMarker` and `ExtRegistryEntry::LoopHeader`
//! variants stay anchored on the [`crate::translator::rtyper::extregistry`]
//! enum so `lookup` / `makekey` / `is_registered` go through the same
//! registry as every other extregistry subclass; `compute_result_annotation`
//! itself, however, is upstream-defined in `rlib/jit.py` and lives here
//! to keep the file boundary aligned with upstream.
//!
//! The user-facing no-op hint functions from the earlier half of
//! `rlib/jit.py` (`promote`, `promote_string`, `we_are_jitted`,
//! `record_exact_class`, and related helpers) live in
//! `majit_metainterp::jit`. The proc macro and interpreter examples
//! recognize that public path today, while this crate owns the
//! translator-only extregistry half.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::annotator::bookkeeper::{Bookkeeper, EmulatedPbcCallKey};
use crate::annotator::model::{AnnotatorError, SomeValue, s_none};
use crate::flowspace::model::{ConstValue, HostObject};

/// Upstream `jit.py:889` — `self.instance.__name__` on the bound
/// method. Discriminates which member of the
/// `_about_ = (jit_merge_point, can_enter_jit)` tuple triggered the
/// `ExtEnterLeaveMarker` entry; Rust passes the kind explicitly at
/// registration / lookup time because it has no bound-method object
/// identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JitMarkerKind {
    /// `jit_merge_point` (jit.py:756-759).
    JitMergePoint,
    /// `can_enter_jit` (jit.py:761-766).
    CanEnterJit,
}

impl JitMarkerKind {
    /// Upstream `self.instance.__name__` — the bound-method name for
    /// `(jit_merge_point, can_enter_jit)` registered via `_about_` at
    /// `rlib/jit.py:798-810 _make_extregistryentries`. Used by
    /// [`ext_enter_leave_marker_compute_result_annotation`] to gate the
    /// `annotate_hooks` branch (`rlib/jit.py:889`) and as the
    /// `methodname` carried on the [`crate::annotator::model::SomeBuiltin`]
    /// returned by the marker's `compute_annotation` (mirroring upstream
    /// `extregistry.py:62-67` base implementation).
    pub fn upstream_method_name(self) -> &'static str {
        match self {
            JitMarkerKind::JitMergePoint => "jit_merge_point",
            JitMarkerKind::CanEnterJit => "can_enter_jit",
        }
    }

    /// Diagnostic analyser-name for the `SomeBuiltin` carrier returned
    /// by `ExtEnterLeaveMarker.compute_annotation`. Not registered in
    /// `BUILTIN_ANALYZERS` — marker dispatch is special-cased in
    /// `SomeValue::call` via `extregistry::lookup(const_box)` rather
    /// than going through the analyser-name table (see
    /// `annotator/model.rs` SomeValue::Builtin call arm).
    pub fn analyser_name(self) -> &'static str {
        match self {
            JitMarkerKind::JitMergePoint => "rlib.jit.ExtEnterLeaveMarker.jit_merge_point",
            JitMarkerKind::CanEnterJit => "rlib.jit.ExtEnterLeaveMarker.can_enter_jit",
        }
    }
}

/// `rlib/jit.py:1018` — `self.instance.__name__` for `loop_header`.
pub const LOOP_HEADER_METHOD_NAME: &str = "loop_header";

/// Diagnostic analyser-name for `ExtLoopHeader.compute_annotation`'s
/// `SomeBuiltin` carrier; same special-cased dispatch as the
/// enter/leave marker (see [`JitMarkerKind::analyser_name`]).
pub const LOOP_HEADER_ANALYSER_NAME: &str = "rlib.jit.ExtLoopHeader.loop_header";

/// Translator-side mirror of the runtime `PyPyJitDriver` (eval.rs)
/// fields that `ExtEnterLeaveMarker` / `ExtLoopHeader` consume. RPython
/// keeps a single `JitDriver` instance reachable from
/// `ExtRegistryEntry.instance.im_self` (rlib/jit.py:892) — the Rust
/// port stores the same fields here behind an `Arc` so multiple
/// registry variants for the same driver share the metadata identity.
///
/// The metadata is intentionally narrow: only fields that the
/// annotator / codewriter pipeline reads. Runtime hooks captured on
/// the pyre side (`PyPyJitDriver::get_*`) are mirrored as `HostObject`
/// callables once the per-hook wiring lands in S1.3.
#[derive(Clone, Debug)]
pub struct JitDriverMeta {
    /// Identity of the driver instance — used as the
    /// `_jit_annotation_cache` key (rlib/jit.py:904
    /// `cache = self.bookkeeper._jit_annotation_cache[driver]`).
    pub id: HostObject,
    /// rlib/jit.py:619 / 651 — `name`.
    pub name: String,
    /// rlib/jit.py:649-650 — `greens`. Order is significant
    /// (rlib/jit.py:736-744 INT/REF/FLOAT heuristic check).
    pub greens: Vec<String>,
    /// rlib/jit.py:652-662 — `reds`.
    pub reds: Vec<String>,
    /// rlib/jit.py:665-668 — `virtualizables`.
    pub virtualizables: Vec<String>,
    /// rlib/jit.py:653 / 661 — True iff `reds='auto'`.
    pub autoreds: bool,
    /// rlib/jit.py:655 / 662 — `len(reds)`; `None` when `autoreds`.
    pub numreds: Option<usize>,
    /// rlib/jit.py:692.
    pub is_recursive: bool,
    /// rlib/jit.py:693 — `vec` (vectorize default False).
    pub vec: bool,
    /// rlib/jit.py:682 / 925-929 — `get_printable_location` callable
    /// for `annotate_hooks`. Wired by the pyre side once the host
    /// callable identity is available.
    pub get_printable_location: Option<HostObject>,
    /// rlib/jit.py:683 / 925-929 — `get_location` callable for
    /// `annotate_hooks`.
    pub get_location: Option<HostObject>,
}

/// rlib/jit.py:886-923 — `ExtEnterLeaveMarker.compute_result_annotation`.
///
/// ```python
/// def compute_result_annotation(self, **kwds_s):
///     if self.instance.__name__ == 'jit_merge_point':
///         self.annotate_hooks(**kwds_s)
///     driver = self.instance.im_self
///     keys = kwds_s.keys()
///     keys.sort()
///     expected = ['s_' + name for name in driver.greens + driver.reds
///                             if '.' not in name]
///     expected.sort()
///     if keys != expected:
///         raise JitHintError(...)
///     try:
///         cache = self.bookkeeper._jit_annotation_cache[driver]
///     except (AttributeError, KeyError):
///         cache = {}
///         self.bookkeeper._jit_annotation_cache[driver] = cache
///     for key, s_value in kwds_s.items():
///         s_previous = cache.get(key, annmodel.s_ImpossibleValue)
///         s_value = annmodel.unionof(s_previous, s_value)
///         cache[key] = s_value
///     # add the attribute _dont_reach_me_in_del_ ...
///     return annmodel.s_None
/// ```
pub fn ext_enter_leave_marker_compute_result_annotation(
    bookkeeper: &Rc<Bookkeeper>,
    meta: &Arc<JitDriverMeta>,
    marker_kind: JitMarkerKind,
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // rlib/jit.py:889-890 — `if self.instance.__name__ ==
    // 'jit_merge_point': self.annotate_hooks(**kwds_s)`. Upstream
    // sequences annotate_hooks BEFORE the kwds validation below; mirror
    // the order so a missing-kwarg error from inside annotate_hooks
    // (KeyError on `kwds_s['s_' + name]`) surfaces with the same
    // ordering as upstream.
    if marker_kind == JitMarkerKind::JitMergePoint {
        annotate_hooks(bookkeeper, meta, kwds_s)?;
    }

    // rlib/jit.py:892-901 — validate keyword names against
    // `driver.greens + driver.reds`. Dotted greenfield names
    // (e.g. `'frame.code'`) are excluded from the kwds key set
    // because `specialize_call` projects them through field access
    // rather than through a kwarg.
    let mut expected: Vec<String> = meta
        .greens
        .iter()
        .chain(meta.reds.iter())
        .filter(|name| !name.contains('.'))
        .map(|name| format!("s_{name}"))
        .collect();
    expected.sort();

    let mut actual: Vec<String> = kwds_s.keys().cloned().collect();
    actual.sort();

    if actual != expected {
        return Err(AnnotatorError::new(format!(
            "JitDriver({}) marker expects keyword arguments {expected:?}, got {actual:?}",
            meta.name
        )));
    }

    // rlib/jit.py:903-914 — fold into _jit_annotation_cache[driver].
    bookkeeper.union_jit_annotation_kwds(&meta.id, kwds_s)?;

    // rlib/jit.py:916-921 — `try: graph = self.bookkeeper.position_key[0];
    //                            graph.func._dont_reach_me_in_del_ = True
    //                       except (TypeError, AttributeError): pass`.
    //
    // Rust mirrors the silent-swallow contract by walking the optional
    // chain and giving up at the first miss: position_key absent / weak
    // graph dead / `func` absent.
    if let Some(pk) = bookkeeper.current_position_key()
        && let Some(graph_rc) = pk.graph()
        && let Some(func) = graph_rc.borrow_mut().func.as_mut()
    {
        func._dont_reach_me_in_del_ = true;
    }

    // rlib/jit.py:923 — `return annmodel.s_None`.
    Ok(s_none())
}

/// rlib/jit.py:925-929 — `ExtEnterLeaveMarker.annotate_hooks`.
///
/// ```python
/// def annotate_hooks(self, **kwds_s):
///     driver = self.instance.im_self
///     h = self.annotate_hook
///     h(driver.get_printable_location, driver.greens, **kwds_s)
///     h(driver.get_location, driver.greens, **kwds_s)
/// ```
fn annotate_hooks(
    bookkeeper: &Rc<Bookkeeper>,
    meta: &Arc<JitDriverMeta>,
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<(), AnnotatorError> {
    annotate_hook(
        bookkeeper,
        meta.get_printable_location.as_ref(),
        &meta.greens,
        kwds_s,
    )?;
    annotate_hook(bookkeeper, meta.get_location.as_ref(), &meta.greens, kwds_s)?;
    Ok(())
}

/// rlib/jit.py:931-950 — `ExtEnterLeaveMarker.annotate_hook`.
///
/// ```python
/// def annotate_hook(self, func, variables, args_s=[], **kwds_s):
///     if func is None:
///         return
///     bk = self.bookkeeper
///     s_func = bk.immutablevalue(func)
///     uniquekey = 'jitdriver.%s' % func.__name__
///     args_s = args_s[:]
///     for name in variables:
///         if '.' not in name:
///             s_arg = kwds_s['s_' + name]
///         else:
///             objname, fieldname = name.split('.')
///             s_instance = kwds_s['s_' + objname]
///             classdesc = s_instance.classdef.classdesc
///             bk.record_getattr(classdesc, fieldname)
///             attrdef = s_instance.classdef.find_attribute(fieldname)
///             s_arg = attrdef.s_value
///             assert s_arg is not None
///         args_s.append(s_arg)
///     bk.emulate_pbc_call(uniquekey, s_func, args_s)
/// ```
///
/// The `args_s=[]` upstream default is dropped here because no
/// in-tree caller forwards a non-empty seed; if one ever lands the
/// signature gains an `args_s_seed: &[SomeValue]` parameter without
/// touching the variable-loop body.
///
/// The dotted-green branch is held back to S1.3 alongside
/// `ExtEnterLeaveMarker.specialize_call` (rlib/jit.py:965-993). Until
/// then a dotted variable surfaces as an `AnnotatorError` so the gap
/// fails closed instead of silently dropping the arg.
fn annotate_hook(
    bookkeeper: &Rc<Bookkeeper>,
    func: Option<&HostObject>,
    variables: &[String],
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<(), AnnotatorError> {
    // rlib/jit.py:932-933 — `if func is None: return`.
    let Some(func) = func else { return Ok(()) };

    // rlib/jit.py:935 — `s_func = bk.immutablevalue(func)`. The hook
    // host is wrapped as a `ConstValue::HostObject` so the bookkeeper
    // takes the same branch as upstream's `immutablevalue(func)` for
    // a Python function (returns `SomePBC([funcdesc])`).
    let s_func = bookkeeper.immutablevalue(&ConstValue::HostObject(func.clone()))?;

    // rlib/jit.py:936 — `uniquekey = 'jitdriver.%s' % func.__name__`.
    let unique_key = EmulatedPbcCallKey::Text(format!("jitdriver.{}", func.qualname()));

    // rlib/jit.py:937-949 — build `args_s` from greens.
    let mut args_s: Vec<SomeValue> = Vec::with_capacity(variables.len());
    for name in variables {
        if name.contains('.') {
            // rlib/jit.py:941-948 dotted-green field path. Requires
            // `s_instance.classdef.find_attribute(fieldname)` plus
            // `bk.record_getattr(classdesc, fieldname)` — both land
            // with `ExtEnterLeaveMarker.specialize_call` at S1.3.
            return Err(AnnotatorError::new(format!(
                "JitDriver dotted greenfield {name:?} reaches annotate_hook before \
                 ExtEnterLeaveMarker.specialize_call port (S1.3, rlib/jit.py:965-993)",
            )));
        }
        let key = format!("s_{name}");
        let s_arg = kwds_s.get(&key).ok_or_else(|| {
            // rlib/jit.py:940 raises KeyError on `kwds_s['s_'+name]`.
            AnnotatorError::new(format!(
                "JitDriver hook annotate: missing kwarg {key:?} for green {name:?}"
            ))
        })?;
        // upstream pushes the kwds entry to args_s as-is; if the entry
        // is `None` (annotation not yet bound at this fixpoint round),
        // `emulate_pbc_call(args_s)` would AttributeError on the first
        // `s.knowntype` access downstream.  `emulate_pbc_call` still
        // takes `&[SomeValue]` so we surface the unbound-cell case as
        // an `AnnotatorError` here; widening that ABI to
        // `&[Option<SomeValue>]` is a follow-up.
        let s_arg = s_arg.as_ref().ok_or_else(|| {
            AnnotatorError::new(format!(
                "JitDriver hook annotate: kwarg {key:?} for green {name:?} \
                 is not yet annotated"
            ))
        })?;
        args_s.push(s_arg.clone());
    }

    // rlib/jit.py:950 — `bk.emulate_pbc_call(uniquekey, s_func, args_s)`.
    // Upstream defaults: `replace=[]`, `callback=None`.
    bookkeeper.emulate_pbc_call(unique_key, &s_func, &args_s, &[], None)?;
    Ok(())
}
