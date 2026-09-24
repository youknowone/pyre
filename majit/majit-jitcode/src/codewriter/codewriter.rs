//! What `pyjitpl.py` `MetaInterpStaticData.finish_setup(codewriter)` reads
//! off the codewriter.

use majit_ir::CallInfoCollection;

use super::assembler::Assembler;
use super::call::VirtualRefInfoHandle;

/// The `codewriter` argument of `finish_setup`: `codewriter.assembler` and
/// the three `codewriter.callcontrol` fields it copies.
pub trait CodeWriterSetup {
    /// `codewriter.assembler`.
    fn assembler(&self) -> &Assembler;
    /// `codewriter.callcontrol.virtualref_info`.
    fn virtualref_info(&self) -> Option<&std::sync::Arc<dyn VirtualRefInfoHandle>>;
    /// `codewriter.callcontrol.callinfocollection`.
    fn callinfocollection(&self) -> &CallInfoCollection;
    /// `codewriter.callcontrol.has_libffi_call`.
    fn has_libffi_call(&self) -> bool;
}
