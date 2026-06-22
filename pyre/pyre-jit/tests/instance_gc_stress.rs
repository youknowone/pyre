//! End-to-end check that `W_InstanceObject` is GC-managed: an instance's
//! movable attribute values (list / str / dict, reachable only through
//! the instance's mapdict storage slots), a devolved instance's stored
//! values (reached via the storage back-edge), and a materialised
//! `__dict__` view all survive repeated full collections, while dead
//! throwaway instances are reclaimed each round.
//!
//! The harness mirrors the `pyrex` launcher (`pyrex/src/lib.rs`
//! `real_main` + `run_source`) exactly: it does NOT build the GC up
//! front. The program's module body and `run()` use `while` loops (no
//! `FOR_ITER`), so eval reaches a JIT-eligible frame and builds the GC
//! lazily — matching production, where builtins already exist as
//! immortal objects before the GC comes up. `gc.collect()` then forces a
//! deterministic full collection through the collect hook
//! (`interp_gc.py:7-26 collect`).
//!
//! Non-vacuity is asserted AFTER eval: the stable instance allocator
//! hook must be live, proving the GC was actually built during the run
//! and instances therefore routed through it (rather than the leaking
//! `lltype::malloc` Box fallback, which would make the survival checks
//! meaningless).

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

// `run()` is a `while`-loop function so eval builds the GC lazily during
// its execution (no `FOR_ITER` anywhere). `a` is a non-devolved instance
// whose list / str / dict attr values live in mapdict storage slots; `b`
// devolves past the attribute limit, with its `__dict__` materialised
// while live (`view`) and kept rooted across the collections. Each round
// allocates fresh garbage and a dead throwaway instance. The returned
// checksum is reachable only if every live value survived the 100
// collections.
const PROGRAM: &str = r#"
import gc

class A:
    pass

class B:
    pass

def run():
    a = A()
    a.lst = [1, 2, 3, 4, 5]
    a.s = "hello" * 10
    a.d = {"k": [9, 8, 7]}

    b = B()
    i = 0
    while i < 85:
        setattr(b, "f%d" % i, [i, i + 1, i + 2])
        i = i + 1
    view = b.__dict__

    n = 0
    while n < 100:
        junk = [0] * 50
        tmp = A()
        tmp.q = [7, 7]
        gc.collect()
        n = n + 1

    total = a.lst[0] + a.lst[4] + len(a.s) + a.d["k"][2]
    i = 0
    while i < 85:
        v = getattr(b, "f%d" % i)
        total = total + v[0] + v[1] + v[2]
        i = i + 1
    total = total + len(view) + view["f0"][0]
    return total

result = run()
assert result == 11113, result
"#;

fn run_harness() -> Result<(), String> {
    // Mirror `pyrex::real_main` startup, then `pyrex::run_source`.
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    init_jit_hooks();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd);
    importing::set_sys_argv(&["<instance_gc_stress>".to_string()]);

    let code = compile_source_with_filename(PROGRAM, Mode::Exec, "<instance_gc_stress>")
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
    // successful return means every read-back assertion held. The GC is
    // built lazily inside this call (the module frame and `run()` are
    // `FOR_ITER`-free), exactly as in the launcher.
    eval_with_jit(&mut frame).map_err(|e| format!("execution error: {}", e.message))?;

    // Non-vacuity: the stable instance allocator hook is installed by the
    // `JIT_DRIVER` initializer (`driver_pair` → `set_gc_allocator`). If it
    // is live now, the GC was built during eval, so `w_instance_new`
    // routed instances through it rather than the leaking Box fallback —
    // the survival checks above were meaningful.
    let probe = pyre_object::try_gc_alloc_stable(
        pyre_object::W_INSTANCE_GC_TYPE_ID,
        pyre_object::W_INSTANCE_OBJECT_SIZE,
    )
    .ok_or("GC was not built during eval; instance survival checks would be vacuous")?;
    if probe.is_null() {
        return Err("stable GC alloc hook returned null for an instance-sized block".to_string());
    }
    // The probe block is never rooted; zero it so any later sweep reads a
    // well-formed (null map/storage) payload before reclaiming it.
    unsafe {
        std::ptr::write_bytes(probe, 0, pyre_object::W_INSTANCE_OBJECT_SIZE);
    }
    Ok(())
}

#[test]
fn instance_attrs_survive_full_collection() {
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
        .expect("instance gc stress program failed");
}
