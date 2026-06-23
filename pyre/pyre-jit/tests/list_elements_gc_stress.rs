//! End-to-end check that an object-strategy `W_ListObject`'s elements are
//! GC-traced: instances reachable ONLY through a list survive repeated
//! full collections. Regression for the `W_LIST_GC_TYPE_ID` registration
//! that traced `items` as a single non-managed pointer (the `std::alloc`'d
//! `ItemsBlock`) and never reached the elements — a major collection then
//! swept a list element reachable only via the list. Fixed by giving
//! `W_ListObject` a custom trace (`list_object_custom_trace`) that walks
//! the off-GC block, mirroring `W_TupleObject` / `W_SetObject`.
//!
//! The harness mirrors the `pyrex` launcher exactly: it does NOT build the
//! GC up front. The program is `FOR_ITER`-free (only `while` loops) so eval
//! reaches a JIT-eligible frame and builds the GC lazily, matching
//! production. `gc.collect()` then forces a deterministic full collection.
//!
//! Non-vacuity is asserted AFTER eval: the stable instance allocator hook
//! must be live, proving the GC was built during the run so the list
//! elements were genuinely GC-managed (not the leaking `lltype::malloc`
//! Box fallback, which would make the survival checks meaningless).

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

// `objs` (a list literal → Object strategy) and `grown` (built by
// `append`, exercising object-strategy growth) hold `Node` instances that
// are reachable ONLY through their list. Each round allocates fresh
// nursery garbage and forces a full collection; the list elements must be
// marked through the list's custom trace. The returned checksum is
// reachable only if every element survived the 100 collections.
const PROGRAM: &str = r#"
import gc

class Node:
    pass

def run():
    objs = [Node(), Node(), Node()]
    objs[0].v = 10
    objs[1].v = 20
    objs[2].v = 30

    grown = []
    i = 0
    while i < 12:
        e = Node()
        e.v = i
        grown.append(e)
        i = i + 1

    n = 0
    while n < 100:
        junk = [0] * 64
        gc.collect()
        n = n + 1

    total = objs[0].v + objs[1].v + objs[2].v
    i = 0
    while i < 12:
        total = total + grown[i].v
        i = i + 1
    return total

result = run()
assert result == 126, result
"#;

fn run_harness() -> Result<(), String> {
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    init_jit_hooks();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd);
    importing::set_sys_argv(&["<list_elements_gc_stress>".to_string()]);

    let code = compile_source_with_filename(PROGRAM, Mode::Exec, "<list_elements_gc_stress>")
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

    // Non-vacuity: the stable instance allocator hook is installed by the
    // `JIT_DRIVER` initializer. If it is live now, the GC was built during
    // eval, so the list and its `Node` elements routed through it rather
    // than the leaking Box fallback — the survival checks were meaningful.
    let probe = pyre_object::try_gc_alloc_stable(
        pyre_object::W_OBJECT_OBJECT_GC_TYPE_ID,
        pyre_object::W_OBJECT_OBJECT_SIZE,
    )
    .ok_or("GC was not built during eval; list element survival checks would be vacuous")?;
    if probe.is_null() {
        return Err("stable GC alloc hook returned null for an instance-sized block".to_string());
    }
    unsafe {
        std::ptr::write_bytes(probe, 0, pyre_object::W_OBJECT_OBJECT_SIZE);
    }
    Ok(())
}

#[test]
fn list_elements_survive_full_collection() {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_harness)
        .expect("spawn worker thread");
    handle
        .join()
        .expect("worker thread panicked")
        .expect("list elements gc stress program failed");
}
