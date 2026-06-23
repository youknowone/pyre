//! End-to-end check that a heap type's namespace is GC-rooted: a user
//! class's method (a function), its class attribute (a movable list), and
//! the per-type `__dict__` getset descriptor (whose `fget` is a
//! collectable function) all survive repeated full collections, even when
//! first reached *fresh* after the collections.
//!
//! Heap type objects (`w_type_new`) are Box-immortal, so the collector
//! never fires their `W_TYPE_GC_TYPE_ID` custom trace and never reaches
//! the movable values bound in the type's namespace `DictStorage`.  Before
//! the `HEAP_TYPE_REGISTRY` / `walk_type_dicts_gc` / `walk_raw_getset_roots`
//! root walk, this program SIGSEGV'd: a method call, a class-attribute
//! read, or a fresh `obj.__dict__` access after a `gc.collect()` reached a
//! relocated value or a freed getset getter.
//!
//! The harness mirrors the `pyrex` launcher (`real_main` + `run_source`)
//! exactly: it does NOT build the GC up front.  The module body and
//! `run()` use `while` loops (no `FOR_ITER`), so eval reaches a
//! JIT-eligible frame and builds the GC lazily — matching production.
//! `gc.collect()` then forces a deterministic full collection.
//!
//! Non-vacuity is asserted AFTER eval: the stable allocator hook must be
//! live, proving the GC was actually built during the run.

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

// `run()` is a `while`-loop function so eval builds the GC lazily during
// its execution.  `C` is a user heap type whose namespace dict holds a
// method (`method`), a class attribute (`KLASS_ATTR`), and — once
// `c.__dict__` is first read — the copied `__dict__` getset descriptor.
// Each loop round allocates fresh garbage and a dead throwaway instance
// and forces a full collection.  The returned checksum is reachable only
// if every namespace value (and the descriptor's getter) survived.
const PROGRAM: &str = r#"
import gc

class C:
    KLASS_ATTR = [10, 20, 30]
    def method(self):
        return 7

def run():
    c = C()
    c.x = 5
    n = 0
    while n < 100:
        junk = [0] * 50
        tmp = C()
        gc.collect()
        n = n + 1
    total = c.method()
    total = total + C.KLASS_ATTR[1]
    d = c.__dict__
    total = total + len(d) + d["x"]
    return total

result = run()
assert result == 33, result
"#;

fn run_harness() -> Result<(), String> {
    // Mirror `pyrex::real_main` startup, then `pyrex::run_source`.
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    init_jit_hooks();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd);
    importing::set_sys_argv(&["<type_namespace_gc_stress>".to_string()]);

    let code = compile_source_with_filename(PROGRAM, Mode::Exec, "<type_namespace_gc_stress>")
        .map_err(|e| format!("compile error: {e}"))?;

    register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    set_build_class_exec_ctx(Rc::as_ptr(&execution_context));
    set_last_exec_ctx(Rc::as_ptr(&execution_context));

    let mut frame = PyFrame::new_with_context(code, execution_context)
        .map_err(|e| format!("frame setup error: {}", e.message))?;

    // Reuse the canonical globals dict as the __main__ module's dict so
    // `globals()` / `function.__globals__` share one identity
    // (`run_source` parity).
    let canonical = frame.get_w_globals();
    let main_module = pyre_object::w_module_new_aliasing_dict(
        "__main__",
        unsafe { pyre_object::w_dict_get_dict_storage_proxy(canonical) },
        canonical,
    );
    importing::set_sys_module("__main__", main_module);

    // An uncaught `assert` in the program surfaces here as `Err`, so a
    // successful return means every read-back assertion held.  The GC is
    // built lazily inside this call (the module frame and `run()` are
    // `FOR_ITER`-free), exactly as in the launcher.
    eval_with_jit(&mut frame).map_err(|e| format!("execution error: {}", e.message))?;

    // Non-vacuity: the stable allocator hook is installed by the
    // `JIT_DRIVER` initializer (`driver_pair` -> `set_gc_allocator`).  If it
    // is live now, the GC was built during eval, so the namespace values
    // above were actually subject to relocation/reclamation — the survival
    // checks were meaningful.
    let probe = pyre_object::try_gc_alloc_stable(
        pyre_object::W_OBJECT_OBJECT_GC_TYPE_ID,
        pyre_object::W_OBJECT_OBJECT_SIZE,
    )
    .ok_or("GC was not built during eval; namespace survival checks would be vacuous")?;
    if probe.is_null() {
        return Err("stable GC alloc hook returned null".to_string());
    }
    unsafe {
        std::ptr::write_bytes(probe, 0, pyre_object::W_OBJECT_OBJECT_SIZE);
    }
    Ok(())
}

#[test]
fn type_namespace_survives_full_collection() {
    // Mirror the launcher's 256 MiB worker stack so deep tracer /
    // interpreter recursion does not overflow the default test stack. GC
    // hooks are thread-local, so the whole harness runs on this thread.
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_harness)
        .expect("spawn worker thread");
    handle
        .join()
        .expect("worker thread panicked")
        .expect("type namespace gc stress program failed");
}
