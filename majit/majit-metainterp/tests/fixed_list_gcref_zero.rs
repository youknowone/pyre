//! `FixedSignedList` R banks are `GcArray(GCREF)`. `malloc_fast` writes
//! only the tid; `ll_newlist` / `gct_zero_gc_pointers_inside` null every
//! item so a collection that reaches the array does not trace leftover
//! nursery words as pointers.

use majit_backend::jitframe::jitframe_type_info;
use majit_gc::GcAllocator;
use majit_gc::collector::MiniMarkGC;
use majit_gc::gc_sync;
use majit_gc::trace::TypeInfo;
use majit_metainterp::install_active_backend_gc_standalone;
use majit_metainterp::opencoder::{FixedSignedList, register_trace_ops_gc_type};

fn install() {
    gc_sync::store_singleton(Box::new(MiniMarkGC::new()));
    majit_gc::shadow_stack::register_mutator();
    gc_sync::register_thread();
    gc_sync::gc_op(|gc| {
        let jitframe_tid = gc.register_type(jitframe_type_info());
        gc.set_jitframe_type_id(jitframe_tid);
        register_trace_ops_gc_type(gc);
    });
    install_active_backend_gc_standalone();
}

#[test]
fn r_kind_gcref_items_are_null_after_nursery_reuse() {
    install();
    gc_sync::gc_op(|gc| {
        let tid = gc.register_type(TypeInfo::simple(64));
        for _ in 0..128 {
            let obj = gc.alloc_nursery_typed(tid, 64);
            if obj.0 != 0 {
                unsafe {
                    std::ptr::write_bytes(obj.0 as *mut u8, 0x11, 64);
                }
            }
        }
        gc.do_collect_nursery();
    });

    let mut list = FixedSignedList::alloc_with_roots(4, true, &mut []);
    assert_eq!(
        &list[..],
        &[0, 0, 0, 0],
        "gct_zero_gc_pointers_inside: leftover nursery words must not be live GCREF items"
    );
    let root = majit_gc::shadow_stack::OwnerRootGuard::new(list.as_gcref());
    gc_sync::gc_op(|gc| gc.do_collect_nursery());
    list.reload(root.get());
    assert_eq!(&list[..], &[0, 0, 0, 0]);
}
