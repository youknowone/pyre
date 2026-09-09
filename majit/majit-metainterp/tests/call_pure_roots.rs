//! `util.py args_dict` / `history.py ConstPtr`: real moving-GC coverage.
//! A separate process isolates the active collector and identity-hash hooks.

use majit_gc::{GcAllocator, TypeInfo, collector::MiniMarkGC, gc_sync, shadow_stack};
use majit_ir::{GcRef, Value};
use majit_metainterp::optimizeopt::{optimizer::Optimizer, util::args_dict};

fn root_count() -> usize {
    let mut count = 0;
    shadow_stack::walk_roots(|_| count += 1);
    count
}

fn as_ref(value: Option<Value>) -> GcRef {
    match value {
        Some(Value::Ref(root)) => root,
        other => panic!("expected cached Ref, got {other:?}"),
    }
}

#[test]
fn cache_constants_survive_movement_handoff_and_retire_with_the_last_owner() {
    let mut gc = MiniMarkGC::new();
    let tid = gc.register_type(TypeInfo::simple(16));
    gc_sync::store_singleton(Box::new(gc));
    shadow_stack::register_mutator();
    gc_sync::register_thread();
    majit_gc::set_active_gc_id_or_identityhash(Some(|addr| {
        gc_sync::gc_op(|gc| gc.id_or_identityhash(addr))
    }));
    let initial_roots = root_count();

    let key = gc_sync::gc_op(|gc| gc.alloc_with_type(tid, 16));
    let result = gc_sync::gc_op(|gc| gc.alloc_with_type(tid, 16));
    unsafe {
        *(key.0 as *mut usize) = 42;
        *(result.0 as *mut usize) = 99;
    }
    let cache = args_dict();
    cache.insert(vec![Value::Int(7)], Value::Ref(result));
    cache.insert(vec![Value::Ref(key)], Value::Ref(result));
    cache.insert(vec![Value::Int(8)], Value::Ref(key));
    let stable_hash = majit_gc::gc_id_or_identityhash(key.0);
    assert_eq!(root_count(), initial_roots + 4);
    let mut optimizer = Optimizer::new();
    optimizer.call_pure_results = cache.clone();
    let compile_owner = cache.clone();
    drop(cache);
    assert_eq!(
        root_count(),
        initial_roots + 4,
        "handoff must share Const roots"
    );

    // Neither raw local pointer is a root: only the dictionary retains them.
    gc_sync::gc_op(|gc| gc.collect_generation(-1));
    let moved_key = as_ref(compile_owner.get(&[Value::Int(8)]));
    let moved_result = as_ref(optimizer.get_call_pure_result(&[Value::Int(7)]));
    assert_ne!(moved_key, key);
    assert_ne!(moved_result, result);
    assert_eq!(majit_gc::gc_id_or_identityhash(moved_key.0), stable_hash);
    assert_eq!(
        compile_owner.get(&[Value::Ref(moved_key)]),
        Some(Value::Ref(moved_result))
    );
    assert_eq!(unsafe { *(moved_key.0 as *const usize) }, 42);
    assert_eq!(unsafe { *(moved_result.0 as *const usize) }, 99);

    // Equal keys after forwarding replace the original entry, not a duplicate
    // under a new address hash. Both compiler and optimizer see the update.
    compile_owner.insert(vec![Value::Ref(moved_key)], Value::Int(123));
    assert_eq!(
        optimizer.get_call_pure_result(&[Value::Ref(moved_key)]),
        Some(Value::Int(123))
    );
    assert_eq!(root_count(), initial_roots + 3);
    drop(compile_owner);
    assert_eq!(root_count(), initial_roots + 3);
    drop(optimizer);
    assert_eq!(root_count(), initial_roots);
    majit_gc::set_active_gc_id_or_identityhash(None);
    shadow_stack::unregister_mutator();
    gc_sync::unregister_thread();
}
