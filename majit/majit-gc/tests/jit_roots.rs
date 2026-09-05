//! The embedding collector must see live slots, not copies of their values.
use majit_gc::shadow_stack as roots;
use majit_ir::GcRef;
use std::sync::atomic::{AtomicUsize, Ordering};

static DEADFRAME: AtomicUsize = AtomicUsize::new(0);

unsafe fn trace_frame(addr: usize, visit: &mut dyn FnMut(*mut GcRef)) {
    visit(addr as *mut GcRef);
}

fn walk_deadframe(visit: &mut dyn FnMut(usize)) {
    visit(DEADFRAME.load(Ordering::Relaxed));
}

#[test]
fn embedding_walk_forwards_stack_blackhole_resume_and_deadframe_slots() {
    // This integration-test process owns the once-installed tracer.
    roots::register_libc_jitframe_tracer(trace_frame);
    let mut frame = Box::new(GcRef(0x2000));
    let frame_addr = (&mut *frame as *mut GcRef) as usize;
    roots::register_libc_jitframe(frame_addr);
    let jf_depth = roots::push_jf(GcRef(frame_addr));
    let stack = roots::OwnerRootGuard::new(GcRef(0x1000));
    let mut regs = [0x3000_i64];
    let mut tmp = 0x4000_i64;
    let mut exc = 0x5000_i64;
    let bh_depth = unsafe { roots::push_bh_regs(&mut regs, &mut tmp, &mut exc) };
    let mut resume = [0x6000_i64];
    let resume_depth = roots::resume_ref_roots_depth();
    unsafe { roots::push_resume_ref_roots(&mut resume) };
    let mut deadframe = Box::new(GcRef(0x7000));
    DEADFRAME.store((&mut *deadframe as *mut GcRef) as usize, Ordering::Relaxed);
    majit_gc::set_active_gc_deadframe_hooks(majit_gc::ActiveGcDeadFrameHooks {
        walk_live_deadframes: Some(walk_deadframe),
    });

    roots::walk_jit_roots(|slot| {
        // A separate embedding heap owns these seven references, not jitframes.
        if (0x1000..=0x7000).contains(&slot.0) {
            slot.0 += 0x10000;
        }
    });

    majit_gc::set_active_gc_deadframe_hooks(Default::default());
    roots::pop_resume_ref_roots_to(resume_depth);
    roots::pop_bh_regs_to(bh_depth);
    roots::pop_jf_to(jf_depth);
    roots::unregister_libc_jitframe(frame_addr);
    assert_eq!(stack.get(), GcRef(0x11000));
    assert_eq!(*frame, GcRef(0x12000));
    assert_eq!(regs, [0x13000]);
    assert_eq!(tmp, 0x14000);
    assert_eq!(exc, 0x15000);
    assert_eq!(resume, [0x16000]);
    assert_eq!(*deadframe, GcRef(0x17000));
}
