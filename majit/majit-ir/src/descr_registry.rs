//! Process-global descriptor registry — thin facade over [`crate::descr::gc_cache`].
//!
//! PyPy `pyjitpl.py finish_setup_descrs`:
//!
//! ```python
//! def finish_setup_descrs(self):
//!     from rpython.jit.codewriter import effectinfo
//!     self.all_descrs = self.cpu.setup_descrs()
//!     effectinfo.compute_bitstrings(self.all_descrs)
//! ```
//!
//! `cpu.setup_descrs()` walks `gc_cache._cache_*` (`backend/llsupport/descr.py`)
//! across the six categories — size / field / array / arraylen / call /
//! interiorfield — and returns the concatenated descr list in fixed
//! group order.  Pyre's lift routes every mint site through the
//! process-global [`crate::descr::gc_cache`] (lift of PyPy's per-CPU
//! `gc_ll_descr.gc_cache`); this module is a backwards-compatible
//! `register_*` / `snapshot_*` shim so callers that pre-date the
//! `gc_cache.get_*_descr(LLType, ...)` cache-or-mint API can still
//! publish their freshly-minted descrs.
//!
//! TODO: this module is the temporary surface for
//! mint sites that bypass the keyed cache.  Each migrated site drops
//! its `register_*` call in favour of `gc_cache.get_*_descr(LLType, ...)`;
//! the end state retires this module entirely once every production
//! mint flows through the keyed path.

use crate::descr::{DescrRef, gc_cache};

/// `descr.py get_size_descr` cache-miss publication — register
/// a freshly-minted size descr.  TODO: prefer
/// `gc_cache.get_size_descr(LLType::Struct(...), ...)`.
pub fn register_size(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_size(descr);
}

/// Keyed sibling: publishes the descr to `gc_cache._cache_size[key]`
/// AND `_cache_size_order`, so subsequent
/// `gc_cache.get_size_descr(key, ...)` calls return the same Arc.
/// Mirrors `descr.py get_size_descr` cache-miss
/// `cache[STRUCT] = sizedescr` (the keyed half) for mint sites that
/// bypass `get_size_descr` proper.
pub fn register_keyed_size(key: crate::descr::LLType, descr: DescrRef) {
    gc_cache().lock().unwrap().register_keyed_size(key, descr);
}

/// `descr.py get_field_descr` cache-miss publication.
/// TODO: prefer
/// `gc_cache.get_field_descr(LLType::Struct(...), ...)`.
pub fn register_field(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_field(descr);
}

/// Keyed sibling: publishes the descr to
/// `gc_cache._cache_field[struct_key][field_name]` AND
/// `_cache_field_order`.  Mirrors `descr.py get_field_descr`
/// cache-miss `cachedict[fieldname] = fielddescr` for mint sites that
/// bypass `get_field_descr` proper.
pub fn register_keyed_field(
    struct_key: crate::descr::LLType,
    field_name: String,
    descr: std::sync::Arc<crate::descr::SimpleFieldDescr>,
) {
    gc_cache()
        .lock()
        .unwrap()
        .register_keyed_field(struct_key, field_name, descr);
}

/// `descr.py get_array_descr` cache-miss publication.
/// TODO: prefer
/// `gc_cache.get_array_descr(LLType::Array(...), ...)`.
pub fn register_array(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_array(descr);
}

/// Keyed sibling: publishes the descr to `gc_cache._cache_array[key]`
/// AND `_cache_array_order`.  Mirrors `descr.py get_array_descr`
/// cache-miss `cache[ARRAY_OR_STRUCT] = arraydescr`.
pub fn register_keyed_array(key: crate::descr::LLType, descr: DescrRef) {
    gc_cache().lock().unwrap().register_keyed_array(key, descr);
}

/// `descr.py:374-385 get_arraylen_descr` cache-miss publication.
/// TODO: prefer
/// `gc_cache.get_field_arraylen_descr(LLType::Array(...), ...)`.
pub fn register_array_len(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_arraylen(descr);
}

/// Keyed sibling: publishes the descr to
/// `gc_cache._cache_arraylen[key]` AND `_cache_arraylen_order`.
pub fn register_keyed_arraylen(key: crate::descr::LLType, descr: DescrRef) {
    gc_cache()
        .lock()
        .unwrap()
        .register_keyed_arraylen(key, descr);
}

/// `descr.py get_interiorfield_descr` cache-miss publication.
/// TODO: prefer the keyed cache-or-mint path once
/// `gc_cache.get_interiorfield_descr` lands.
pub fn register_interior_field(descr: DescrRef) {
    gc_cache()
        .lock()
        .unwrap()
        .register_external_interiorfield(descr);
}

/// Keyed sibling: publishes the descr to
/// `gc_cache._cache_interiorfield[(array_key, name, arrayfieldname)]`
/// AND `_cache_interiorfield_order`.  Mirrors `descr.py:404-433
/// get_interiorfield_descr` cache-miss
/// `cache[(ARRAY, name, arrayfieldname)] = interiorfielddescr`.
/// `arrayfieldname == ""` denotes PyPy `arrayfieldname=None`
/// (the GcArray-of-Structs case, `descr.py:431-432`); a non-empty
/// string denotes the GcStruct-containing-inlined-GcArray case
/// (`descr.py:433-434`).
pub fn register_keyed_interior_field(
    array_key: crate::descr::LLType,
    name: String,
    arrayfieldname: String,
    descr: DescrRef,
) {
    gc_cache()
        .lock()
        .unwrap()
        .register_keyed_interiorfield(array_key, name, arrayfieldname, descr);
}

/// `descr.py setup_descrs` snapshot.  Sole production caller is
/// `MetaInterpStaticData::finish_setup_descrs`.  Call descriptors are
/// owned by `GcCache._cache_call`, so the snapshot is the same six-group
/// sequence as PyPy: size, field, array, arraylen, call, interiorfield.
pub fn snapshot_all() -> Vec<DescrRef> {
    let gc = gc_cache().lock().unwrap();
    let mut out = Vec::with_capacity(
        gc.snapshot_sizes().len()
            + gc.snapshot_fields().len()
            + gc.snapshot_arrays().len()
            + gc.snapshot_arraylens().len()
            + gc.snapshot_calls().len()
            + gc.snapshot_interiorfields().len(),
    );
    out.extend(gc.snapshot_sizes());
    out.extend(gc.snapshot_fields());
    out.extend(gc.snapshot_arrays());
    out.extend(gc.snapshot_arraylens());
    out.extend(gc.snapshot_calls());
    out.extend(gc.snapshot_interiorfields());
    out
}

/// `pyjitpl.py self.all_descrs = self.cpu.setup_descrs()` — the dense
/// list `descr_index` indexes into (`descr.py:28 v.descr_index =
/// len(all_descrs)`), and the list `bridgeopt.py:155
/// metainterp_sd.all_descrs[descr_index]` reads back.
///
/// Upstream stores it on `metainterp_sd` because there is exactly one, built
/// from exactly one `cpu.setup_descrs()`.  Pyre's `GcCache` is instead
/// process-global and `set_descr_index` therefore stamps process-globally, so
/// the list those numbers index has to have the same scope: a front-end that
/// carries more than one `MetaInterpStaticData` (pyre keeps a second one for
/// the tracing walker) would otherwise number the descrs off one object's
/// list while `bridgeopt` indexes another object's empty one.
///
/// `MetaInterpStaticData::all_descrs()` is the accessor; nothing else should
/// reach in here directly.
static ALL_DESCRS: std::sync::LazyLock<std::sync::Mutex<std::sync::Arc<Vec<DescrRef>>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::sync::Arc::new(Vec::new())));

/// Handle to the process-wide `all_descrs` list documented on [`ALL_DESCRS`].
pub fn all_descrs() -> &'static std::sync::Mutex<std::sync::Arc<Vec<DescrRef>>> {
    &ALL_DESCRS
}

/// `descr.py:28-29 _cache_size` snapshot.
pub fn snapshot_sizes() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_sizes()
}

/// `descr.py:30-33 _cache_field` snapshot.
pub fn snapshot_fields() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_fields()
}

/// `descr.py:34-36 _cache_array` snapshot.
pub fn snapshot_arrays() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_arrays()
}

/// `descr.py:37-39 _cache_arraylen` snapshot.
pub fn snapshot_array_lens() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_arraylens()
}

/// `descr.py:43-45 _cache_interiorfield` snapshot.
pub fn snapshot_interior_fields() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_interiorfields()
}

/// Per-category counts for diagnostic asserts.
/// Returns `(sizes, fields, arrays, array_lens, interior_fields)` —
/// the call category lives outside this facade; use
/// `gc_cache().lock().unwrap().category_counts()` for the full tuple.
pub fn category_counts() -> (usize, usize, usize, usize, usize) {
    let (s, f, a, al, _c, ifs) = gc_cache().lock().unwrap().category_counts();
    (s, f, a, al, ifs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descr::SimpleFieldDescr;
    use crate::value::Type;
    use std::sync::Arc;

    fn fresh_field(idx: u32) -> DescrRef {
        Arc::new(SimpleFieldDescr::new(idx, 0, 8, Type::Int, false))
    }

    fn count_arc(haystack: &[DescrRef], needle: &DescrRef) -> usize {
        haystack
            .iter()
            .filter(|descr| Arc::ptr_eq(descr, needle))
            .count()
    }

    /// Same `Arc` clone re-registered via the facade collapses to a
    /// single `_cache_field_order` entry — `Arc::ptr_eq` dedup at the
    /// `gc_cache.register_external_*` level.
    #[test]
    fn dedup_by_arc_identity_within_category() {
        let f = fresh_field(42);
        register_field(f.clone());
        register_field(f.clone());
        let fields = gc_cache().lock().unwrap().snapshot_fields();
        assert_eq!(
            count_arc(&fields, &f),
            1,
            "same Arc should collapse to one entry"
        );
    }

    /// Distinct Arcs that share `descr.index() == 0` (e.g. two
    /// different structs' `index_in_parent = 0` fields at
    /// `call.rs`) must NOT collapse — `Arc::ptr_eq` is by
    /// pointer, not by content.  PyPy `_cache_field` parity:
    /// `(STRUCT, fieldname)` keyed dict.
    #[test]
    fn distinct_arcs_share_index_stay_separate() {
        let f_a = fresh_field(0);
        let f_b = fresh_field(0);
        register_field(f_a.clone());
        register_field(f_b.clone());
        let fields = gc_cache().lock().unwrap().snapshot_fields();
        assert_eq!(count_arc(&fields, &f_a), 1);
        assert_eq!(count_arc(&fields, &f_b), 1);
        assert!(
            fields
                .iter()
                .filter(|descr| Arc::ptr_eq(descr, &f_a) || Arc::ptr_eq(descr, &f_b))
                .count()
                == 2
        );
    }
}
