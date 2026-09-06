use std::sync::Arc;

use majit_backend::{DescrContainer, make_and_attach_done_descrs};
use majit_ir::DescrRef;

#[derive(Default)]
struct Target {
    descrs: [Option<DescrRef>; 5],
}

impl DescrContainer for Target {
    fn set_done_with_this_frame_descr_void(&mut self, descr: DescrRef) {
        self.descrs[0] = Some(descr);
    }
    fn set_done_with_this_frame_descr_int(&mut self, descr: DescrRef) {
        self.descrs[1] = Some(descr);
    }
    fn set_done_with_this_frame_descr_ref(&mut self, descr: DescrRef) {
        self.descrs[2] = Some(descr);
    }
    fn set_done_with_this_frame_descr_float(&mut self, descr: DescrRef) {
        self.descrs[3] = Some(descr);
    }
    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: DescrRef) {
        self.descrs[4] = Some(descr);
    }
}

#[test]
fn targets_share_descriptors_but_separate_runtimes_do_not() {
    let mut first = Target::default();
    let mut second = Target::default();
    let mut separate = Target::default();
    // Exercise the heterogeneous API retained by the metainterpreter.
    let targets: &mut [&mut dyn DescrContainer] = &mut [&mut first, &mut second];
    make_and_attach_done_descrs(targets);
    make_and_attach_done_descrs(&mut [&mut separate]);
    for i in 0..5 {
        let descr = first.descrs[i].as_ref().unwrap();
        assert!(Arc::ptr_eq(descr, second.descrs[i].as_ref().unwrap()));
        assert!(!Arc::ptr_eq(descr, separate.descrs[i].as_ref().unwrap()));
        for j in 0..i {
            assert!(!Arc::ptr_eq(descr, first.descrs[j].as_ref().unwrap()));
        }
    }
}
