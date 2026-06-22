//! End-to-end check that a heap type's `weak_subclasses` list is GC-rooted:
//! a subclass recorded in its base's `weak_subclasses` survives repeated full
//! collections so the base's `mutated()` walk still reaches it.
//!
//! `w_type_add_subclass` stores `w_weakref_new(subclass)` — a `try_gc_alloc`
//! young WEAKREF GcStruct — in the base's off-GC `weak_subclasses: *mut
//! Vec<*mut Weakref>`.  Heap types (`w_type_new`) are Box-immortal, so the
//! collector never fires their `W_TYPE_GC_TYPE_ID` custom trace; the only
//! root walk that reaches a Box-immortal type during collection is
//! `walk_type_dicts_gc`.  Before this fix that walk forwarded `bases` and the
//! namespace values but NOT `weak_subclasses`, so the strong root to the
//! WEAKREF object was missing: the first collection reclaimed it and the
//! base's `weak_subclasses[i]` dangled.  `type.__setattr__` then ran
//! `mutated()` -> `w_type_get_subclasses()` -> `w_weakref_deref()` over the
//! freed slot (a UAF that also dropped the subclass's cache invalidation).
//!
//! Observable: `mutated()` resets each live subclass's cached
//! `compares_by_identity_status` to UNKNOWN.  The program caches `B`'s status
//! as YES (using `B` instances as identity dict keys), churns collections,
//! then defines `A.__eq__`/`A.__hash__` (a `type.__setattr__` that runs
//! `A.mutated('__eq__')`).  If `B`'s weakref survived, its status is reset and
//! fresh `B` keys recompute to "compare by value" -> a 2-key dict collapses to
//! 1 entry.  A dangling weakref leaves the stale YES (or crashes on deref).
//! The harness mirrors `type_namespace_gc_stress.rs`.

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

// `run()` builds the GC lazily (its `while` loops are `FOR_ITER`-free).  `B`
// is defined *inside* `run()` after the GC is live so its `weak_subclasses`
// weakref is a real young GC object (a module-scope `class B(A)` would record
// a null weakref — `w_weakref_new` returns null before the GC is built).
const PROGRAM: &str = r#"
import gc

class A:
    pass

def run():
    n = 0
    while n < 40:
        junk = [0] * 50
        gc.collect()
        n = n + 1
    # GC is live now: B's weakref in A.weak_subclasses is a young GC object.
    class B(A):
        pass
    b1 = B()
    b2 = B()
    seed = {}
    seed[b1] = 1
    seed[b2] = 2                 # distinct by identity -> caches B status = YES
    primed = len(seed)           # 2
    m = 0
    while m < 40:
        junk = [0] * 50
        tmp = A()
        gc.collect()
        m = m + 1
    def beq(self, other):
        return True
    def bhash(self):
        return 7
    A.__eq__ = beq               # type.__setattr__ -> A.mutated('__eq__')
    A.__hash__ = bhash           # must reach B via weak_subclasses
    b3 = B()
    b4 = B()
    d = {}
    d[b3] = 10
    d[b4] = 20                   # compare by value (all equal) -> 1 entry
    return primed * 10 + len(d)

result = run()
assert result == 21, result
"#;

fn run_harness() -> Result<(), String> {
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    init_jit_hooks();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd);
    importing::set_sys_argv(&["<weak_subclasses_gc_stress>".to_string()]);

    let code = compile_source_with_filename(PROGRAM, Mode::Exec, "<weak_subclasses_gc_stress>")
        .map_err(|e| format!("compile error: {e}"))?;

    register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    set_build_class_exec_ctx(Rc::as_ptr(&execution_context));
    set_last_exec_ctx(Rc::as_ptr(&execution_context));

    let mut frame = PyFrame::new_with_context(code, execution_context)
        .map_err(|e| format!("frame setup error: {}", e.message))?;

    let canonical = frame.get_w_globals();
    let main_module = pyre_object::w_module_new_aliasing_dict(
        "__main__",
        unsafe { pyre_object::w_dict_get_dict_storage_proxy(canonical) },
        canonical,
    );
    importing::set_sys_module("__main__", main_module);

    eval_with_jit(&mut frame).map_err(|e| format!("execution error: {}", e.message))?;

    // Non-vacuity: the stable allocator hook is installed by the JIT driver
    // initializer.  If it is live now, the GC was built during eval, so the
    // weakref above was actually subject to relocation/reclamation.
    let probe = pyre_object::try_gc_alloc_stable(
        pyre_object::W_INSTANCE_GC_TYPE_ID,
        pyre_object::W_INSTANCE_OBJECT_SIZE,
    )
    .ok_or("GC was not built during eval; weak_subclasses survival check would be vacuous")?;
    if probe.is_null() {
        return Err("stable GC alloc hook returned null".to_string());
    }
    unsafe {
        std::ptr::write_bytes(probe, 0, pyre_object::W_INSTANCE_OBJECT_SIZE);
    }
    Ok(())
}

#[test]
fn weak_subclasses_survive_full_collection() {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_harness)
        .expect("spawn worker thread");
    handle
        .join()
        .expect("worker thread panicked")
        .expect("weak_subclasses gc stress program failed");
}
