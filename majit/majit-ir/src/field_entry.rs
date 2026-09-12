//! `PreambleOp` sentinel + `FieldEntry` enum stored inside `PtrInfo`
//! struct/array field caches.
//!
//! RPython parity:
//! - `rpython/jit/metainterp/optimizeopt/shortpreamble.py PreambleOp`
//! - `rpython/jit/metainterp/optimizeopt/info.py setfield` —
//!   `_fields[]` element is either a normal Box or a PreambleOp.
//!
//! Pure data; no metainterp deps. Hosted in `majit-ir` so the
//! PtrInfo move that follows can reference these types without a
//! `majit-metainterp → majit-ir` circular dep.

use crate::OpRef;

/// shortpreamble.py: PreambleOp
///
/// Wrapper stored in PtrInfo._fields during Phase 2 import.
/// When `_getfield` (heap.py) encounters this in a field slot,
/// it calls `force_op_from_preamble()` to lazily resolve the value
/// via the short preamble builder.
///
/// RPython stores PreambleOp directly in `_fields[]` (Python's dynamic
/// typing). Rust mirrors this with the `FieldEntry` enum stored in the
/// same `fields` / `items` vectors.
#[derive(Clone, Debug)]
pub struct PreambleOp {
    /// RPython `PreambleOp.op` — the carried Box (= `self.res` from the
    /// short_op), a producer-bound / const [`Operand`](crate::operand::Operand).
    /// For non-invented entries this resolves to the body-visible position
    /// directly; for invented entries (CompoundOp alternates) `op` forwards
    /// to the carried Box via `make_equal_to(source, op)` so resolving `op`
    /// reaches the body-visible position.
    pub op: crate::operand::Operand,
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
    pub same_as_source: Option<crate::operand::Operand>,
}

/// RPython _fields[] element — either a concrete value or a PreambleOp sentinel.
///
/// info.py `setfield` stores either a normal Box or a PreambleOp into
/// `_fields[]`. heap.py `_getfield` checks `isinstance(res, PreambleOp)`
/// to decide whether to force the value via the short preamble.
///
/// Rust equivalent: typed enum instead of Python's duck-typed list.
///
/// Packed to 8 B so `(u32, FieldEntry)` is 16 B. A 4-entry grow of the
/// 24 B pair was 96 B on the regex and/or leaf. Value words reuse
/// [`Operand`]'s tag space (0-5); preamble is an 8-aligned `Box` with
/// tag 6, which `Operand` does not use.
#[repr(transparent)]
pub struct FieldEntry {
    packed: u64,
}

const FE_PREAMBLE: u64 = 6;

impl FieldEntry {
    /// Normal cached field value (info.py setfield).
    #[allow(non_snake_case)]
    pub fn Value(op: crate::operand::Operand) -> Self {
        let packed = op.into_packed();
        FieldEntry { packed }
    }

    /// shortpreamble.py PreambleOp — sentinel stored during Phase 2 import.
    #[allow(non_snake_case)]
    pub fn Preamble(pop: Box<PreambleOp>) -> Self {
        let p = Box::into_raw(pop) as u64;
        debug_assert_eq!(p & 7, 0, "PreambleOp box must be 8-aligned");
        FieldEntry {
            packed: p | FE_PREAMBLE,
        }
    }

    pub fn preamble(pop: PreambleOp) -> Self {
        FieldEntry::Preamble(Box::new(pop))
    }

    fn is_preamble_word(&self) -> bool {
        self.packed != 0 && self.packed & 7 == FE_PREAMBLE
    }

    pub fn as_value(&self) -> Option<crate::operand::Operand> {
        if self.is_preamble_word() {
            None
        } else {
            Some(crate::operand::Operand::clone_from_packed(self.packed))
        }
    }

    pub fn kind(&self) -> FieldEntryKind<'_> {
        if let Some(pop) = self.as_preamble() {
            FieldEntryKind::Preamble(pop)
        } else {
            FieldEntryKind::Value(crate::operand::Operand::clone_from_packed(self.packed))
        }
    }
}

/// View of a [`FieldEntry`] for `match`.
pub enum FieldEntryKind<'a> {
    Value(crate::operand::Operand),
    Preamble(&'a PreambleOp),
}

impl FieldEntry {
    /// Extract the concrete OpRef if this is a `Value` entry.
    /// Returns `None` for `Preamble` entries (those need special handling
    /// via `force_op_from_preamble`).
    pub fn as_opref(&self) -> Option<OpRef> {
        self.as_value().map(|b| b.to_opref())
    }

    /// Returns true if this is a `Preamble` entry.
    pub fn is_preamble(&self) -> bool {
        self.is_preamble_word()
    }

    /// Extract the `PreambleOp` if this is a `Preamble` entry.
    pub fn as_preamble(&self) -> Option<&PreambleOp> {
        if !self.is_preamble_word() {
            return None;
        }
        let p = (self.packed & !7) as *const PreambleOp;
        Some(unsafe { &*p })
    }

    pub fn as_preamble_mut(&mut self) -> Option<&mut PreambleOp> {
        if !self.is_preamble_word() {
            return None;
        }
        let p = (self.packed & !7) as *mut PreambleOp;
        Some(unsafe { &mut *p })
    }

    /// View this slot the same way RPython reads `_fields[]` / `_items[]`
    /// in non-forcing paths such as `serialize_optheap`,
    /// `produce_short_preamble_ops`, and `_expand_infos_from_virtual`.
    ///
    /// Normal values return the stored OpRef. `PreambleOp` entries expose
    /// their original Phase 1 source box (`pop.op`), matching PyPy's
    /// `get_box_replacement(PreambleOp(...))` behavior.
    pub fn as_seen_opref(&self) -> OpRef {
        if let Some(pop) = self.as_preamble() {
            pop.op.to_opref()
        } else {
            crate::operand::Operand::clone_from_packed(self.packed).to_opref()
        }
    }

    /// Box analog of [`as_seen_opref`](Self::as_seen_opref): the carried
    /// box object ([`Operand`](crate::operand::Operand)) rather than its
    /// resolved `OpRef` position. Used where the caller keys a box-identity
    /// map by the field's Phase 1 box — `_expand_infos_from_virtual` (export)
    /// and `setinfo_from_preamble_list` (import) read the same shared virtual
    /// info, so the returned operands coincide by identity (clones of the
    /// same stored handle).
    pub fn as_seen_operand(&self) -> crate::operand::Operand {
        if let Some(pop) = self.as_preamble() {
            pop.op.clone()
        } else {
            crate::operand::Operand::clone_from_packed(self.packed)
        }
    }

    /// Consume and extract the `PreambleOp` if this is a `Preamble` entry.
    pub fn into_preamble(self) -> Option<PreambleOp> {
        if !self.is_preamble_word() {
            return None;
        }
        let p = (self.packed & !7) as *mut PreambleOp;
        std::mem::forget(self);
        Some(*unsafe { Box::from_raw(p) })
    }
}

impl Clone for FieldEntry {
    fn clone(&self) -> Self {
        if self.is_preamble_word() {
            FieldEntry::Preamble(Box::new(self.as_preamble().expect("preamble").clone()))
        } else {
            FieldEntry::Value(crate::operand::Operand::clone_from_packed(self.packed))
        }
    }
}

impl Drop for FieldEntry {
    fn drop(&mut self) {
        if self.is_preamble_word() {
            let p = (self.packed & !7) as *mut PreambleOp;
            drop(unsafe { Box::from_raw(p) });
        } else {
            crate::operand::Operand::drop_packed(self.packed);
        }
    }
}

impl std::fmt::Debug for FieldEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(pop) = self.as_preamble() {
            f.debug_tuple("Preamble").field(pop).finish()
        } else {
            f.debug_tuple("Value")
                .field(&crate::operand::Operand::clone_from_packed(self.packed))
                .finish()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_entry_value_pair_leaves_the_96_byte_class() {
        assert_eq!(
            std::mem::size_of::<FieldEntry>(),
            8,
            "FieldEntry is {} B; must pack into Operand's unused tag",
            std::mem::size_of::<FieldEntry>()
        );
        assert!(
            std::mem::size_of::<(u32, FieldEntry)>() <= 16,
            "(u32, FieldEntry) is {} B; a 4-entry grow must not mint 96 B",
            std::mem::size_of::<(u32, FieldEntry)>()
        );
    }
}
