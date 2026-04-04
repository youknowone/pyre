//! Visitor trait for walking virtual object structures.
//!
//! Provides consistent traversal of all virtual object types during
//! resume data construction, unrolling, and other analyses.
//!
//! Translated from rpython/jit/metainterp/walkvirtual.py.

use majit_ir::{DescrRef, OpRef};

/// walkvirtual.py:4-39: VirtualVisitor
///
/// Abstract base class. Every method raises NotImplementedError in RPython;
/// in Rust this is expressed as required trait methods with no defaults.
/// Implementors must provide all methods — the compiler enforces this
/// at the same strength as RPython's runtime NotImplementedError.
pub trait VirtualVisitor {
    /// walkvirtual.py:5
    fn visit_not_virtual(&mut self, value: OpRef);

    /// walkvirtual.py:8
    fn visit_virtual(&mut self, descr: &DescrRef, fielddescrs: &[DescrRef]);

    /// walkvirtual.py:11
    fn visit_vstruct(&mut self, typedescr: &DescrRef, fielddescrs: &[DescrRef]);

    /// walkvirtual.py:14; info.py call: visitor.visit_varray(self.descr, self._clear)
    fn visit_varray(&mut self, arraydescr: &DescrRef, clear: bool);

    /// walkvirtual.py:17; info.py call: visitor.visit_varraystruct(self.descr, self.getlength(), flddescrs)
    fn visit_varraystruct(
        &mut self,
        arraydescr: &DescrRef,
        length: usize,
        fielddescrs: &[DescrRef],
    );

    /// walkvirtual.py:20
    fn visit_vrawbuffer(
        &mut self,
        func: usize,
        size: usize,
        offsets: &[usize],
        descrs: &[DescrRef],
    );

    /// walkvirtual.py:23
    fn visit_vrawslice(&mut self, offset: usize);

    /// walkvirtual.py:26
    fn visit_vstrplain(&mut self, is_unicode: bool);

    /// walkvirtual.py:29
    fn visit_vstrconcat(&mut self, is_unicode: bool);

    /// walkvirtual.py:32
    fn visit_vstrslice(&mut self, is_unicode: bool);

    /// walkvirtual.py:35
    fn register_virtual_fields(&mut self, virtualbox: OpRef, fieldboxes: &[OpRef]);

    /// walkvirtual.py:38
    fn already_seen_virtual(&mut self, virtualbox: OpRef) -> bool;
}
