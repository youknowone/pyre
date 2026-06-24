//! `PreambleOp` sentinel + `FieldEntry` enum stored inside `PtrInfo`
//! struct/array field caches.
//!
//! RPython parity:
//! - `rpython/jit/metainterp/optimizeopt/shortpreamble.py:11-49 PreambleOp`
//! - `rpython/jit/metainterp/optimizeopt/info.py:203 setfield` —
//!   `_fields[]` element is either a normal Box or a PreambleOp.
//!
//! Pure data; no metainterp deps. Hosted in `majit-ir` so the
//! PtrInfo move that follows can reference these types without a
//! `majit-metainterp → majit-ir` circular dep.

use crate::OpRef;

/// shortpreamble.py:11-49: PreambleOp
///
/// Wrapper stored in PtrInfo._fields during Phase 2 import.
/// When `_getfield` (heap.py:177-187) encounters this in a field slot,
/// it calls `force_op_from_preamble()` to lazily resolve the value
/// via the short preamble builder.
///
/// RPython stores PreambleOp directly in `_fields[]` (Python's dynamic
/// typing). Rust mirrors this with the `FieldEntry` enum stored in the
/// same `fields` / `items` vectors.
#[derive(Clone, Debug)]
pub struct PreambleOp {
    /// RPython `PreambleOp.op` — the carried Box (= `self.res` from the
    /// short_op). For non-invented entries this resolves to the
    /// body-visible position directly; for invented entries (CompoundOp
    /// alternates) `op` forwards to the carried Box via
    /// `make_equal_to(source, op)` so resolving `op` reaches the
    /// body-visible position.
    pub op: crate::box_ref::BoxRef,
    /// RPython: PreambleOp.invented_name
    pub invented_name: bool,
    /// RPython: PreambleOp.preamble_op — the actual replay operation
    /// for the short preamble. Always present (RPython parity).
    pub preamble_op: crate::resoperation::OpRc,
    /// Original result box an invented SameAs name aliases — the
    /// compound-dedup winner's `res`, threaded from
    /// `ProducedShortOp.same_as_source`. Lets an imported pop reproduce the
    /// builder map entry's `same_as(original)` at `add_preamble_op_from_pop`
    /// instead of `same_as(invented_name)` (a self-alias). `None` for
    /// non-invented entries (`invented_name == false`), where the SameAs
    /// arm is never taken.
    pub same_as_source: Option<crate::box_ref::BoxRef>,
}

/// RPython _fields[] element — either a concrete value or a PreambleOp sentinel.
///
/// info.py:203 `setfield` stores either a normal Box or a PreambleOp into
/// `_fields[]`. heap.py:177 `_getfield` checks `isinstance(res, PreambleOp)`
/// to decide whether to force the value via the short preamble.
///
/// Rust equivalent: typed enum instead of Python's duck-typed list.
#[derive(Clone, Debug)]
pub enum FieldEntry {
    /// Normal cached field value (info.py:203 setfield). Stored as an
    /// [`Operand`](crate::operand::Operand) so a `Const` ref is GC-walked
    /// through `Operand::walk_const_ptr_refs`, never persisting a Copy
    /// `OpRef::ConstPtr` that a moving collection cannot reach.
    Value(crate::operand::Operand),
    /// shortpreamble.py:11 PreambleOp — sentinel stored during Phase 2 import.
    Preamble(PreambleOp),
}

impl FieldEntry {
    /// Extract the concrete OpRef if this is a `Value` entry.
    /// Returns `None` for `Preamble` entries (those need special handling
    /// via `force_op_from_preamble`).
    pub fn as_opref(&self) -> Option<OpRef> {
        match self {
            FieldEntry::Value(b) => Some(b.to_opref()),
            FieldEntry::Preamble(_) => None,
        }
    }

    /// Returns true if this is a `Preamble` entry.
    pub fn is_preamble(&self) -> bool {
        matches!(self, FieldEntry::Preamble(_))
    }

    /// Extract the `PreambleOp` if this is a `Preamble` entry.
    pub fn as_preamble(&self) -> Option<&PreambleOp> {
        match self {
            FieldEntry::Preamble(pop) => Some(pop),
            FieldEntry::Value(_) => None,
        }
    }

    /// View this slot the same way RPython reads `_fields[]` / `_items[]`
    /// in non-forcing paths such as `serialize_optheap`,
    /// `produce_short_preamble_ops`, and `_expand_infos_from_virtual`.
    ///
    /// Normal values return the stored OpRef. `PreambleOp` entries expose
    /// their original Phase 1 source box (`pop.op`), matching PyPy's
    /// `get_box_replacement(PreambleOp(...))` behavior.
    pub fn as_seen_opref(&self) -> OpRef {
        match self {
            FieldEntry::Value(b) => b.to_opref(),
            FieldEntry::Preamble(pop) => pop.op.to_opref(),
        }
    }

    /// Box analog of [`as_seen_opref`](Self::as_seen_opref): the carried
    /// `BoxRef` rather than its resolved `OpRef` position. Used where the
    /// caller keys a box-identity (`Rc::ptr_eq`) map by the field's Phase 1
    /// box — `_expand_infos_from_virtual` (export) and
    /// `setinfo_from_preamble_list` (import) read the same shared virtual
    /// info, so the returned boxes coincide by identity. A bound `Value`
    /// operand resolves to its canonical box (memoized on the producer), so
    /// export and import return the same `Rc`; a `Const` operand mints a
    /// fresh box, but consts never enter the `Rc::ptr_eq` map (they carry no
    /// exported info), so the fresh mint is never compared.
    pub fn as_seen_box(&self) -> crate::box_ref::BoxRef {
        match self {
            FieldEntry::Value(b) => b.to_boxref(),
            FieldEntry::Preamble(pop) => pop.op.clone(),
        }
    }

    /// Consume and extract the `PreambleOp` if this is a `Preamble` entry.
    pub fn into_preamble(self) -> Option<PreambleOp> {
        match self {
            FieldEntry::Preamble(pop) => Some(pop),
            FieldEntry::Value(_) => None,
        }
    }
}
