//! End-to-end check that storing a *young* (nursery) value into an
//! object-strategy `W_ListObject` keeps it alive across a minor GC.
//!
//! The list body is old-gen (`try_gc_alloc_stable`); its elements live in
//! an off-GC `ItemsBlock` reached only via `list_object_custom_trace`. A
//! minor (nursery) collection forwards an old-gen container's young refs
//! ONLY when the container sits in the remembered set, which is populated
//! exclusively by the write barrier (`try_gc_write_barrier`). Without a
//! barrier at each ref store, a fresh dict (`{}` is the nursery
//! `W_DICT`) appended/assigned/inserted into a list that is reachable only
//! through that list is never forwarded — the nursery reset leaves a
//! dangling pointer and the later read is a use-after-free.
//!
//! Regression for the missing list write barriers in `w_list_append` /
//! `w_list_setitem` / `w_list_insert` / `w_list_setslice` and
//! `w_list_new_with_strategy` creation. Mirrors `set_write_barrier` /
//! `dict_write_barrier`. Proven non-vacuous: without the barriers the
//! checksum is corrupted / the process faults; with them it is exact.
//!
//! The harness mirrors the `pyrex` launcher: it does NOT build the GC up
//! front. The program is `FOR_ITER`-free (only `while` loops) so eval
//! reaches a JIT-eligible frame and builds the GC lazily, matching
//! production. The nursery churn (`junk` lists + scratch dicts) plus
//! `gc.collect()` forces minor collections during the hot loop.

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

// Each `{}` is a nursery W_DICT reachable only through its list. The four
// list mutators that store a ref — append, literal creation, setitem,
// insert — must each barrier the old-gen list so the next minor GC
// forwards the young dict. The checksum is recoverable only if every
// stored dict survived the 200 collections.
const PROGRAM: &str = r#"
import gc

def run():
    # (1) append: young dicts into an object-strategy list
    appended = []
    i = 0
    while i < 16:
        d = {}
        d['v'] = i
        appended.append(d)
        i = i + 1

    # (2) creation: object-strategy list literal of young dicts
    literal = [{}, {}, {}]
    literal[0]['v'] = 10
    literal[1]['v'] = 20
    literal[2]['v'] = 30

    # (3) setitem + insert of young dicts
    slots = [{}, {}]
    slots[0]['v'] = 1
    slots[1] = {}
    slots[1]['v'] = 2
    slots.insert(1, {})
    slots[1]['v'] = 3

    n = 0
    while n < 200:
        junk = [0] * 32
        scratch = {}
        scratch['x'] = n
        gc.collect()
        n = n + 1

    total = 0
    i = 0
    while i < 16:
        total = total + appended[i]['v']
        i = i + 1
    total = total + literal[0]['v'] + literal[1]['v'] + literal[2]['v']
    total = total + slots[0]['v'] + slots[1]['v'] + slots[2]['v']
    return total

result = run()
assert result == 186, result
"#;

fn run_harness() -> Result<(), String> {
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    init_jit_hooks();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd);
    importing::set_sys_argv(&["<list_young_dict_gc_stress>".to_string()]);

    let code = compile_source_with_filename(PROGRAM, Mode::Exec, "<list_young_dict_gc_stress>")
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
    // eval, so the lists and the young dicts stored into them routed
    // through the real managed heap rather than the leaking Box fallback —
    // the minor-GC survival checks were meaningful.
    let probe = pyre_object::try_gc_alloc_stable(
        pyre_object::W_INSTANCE_GC_TYPE_ID,
        pyre_object::W_INSTANCE_OBJECT_SIZE,
    )
    .ok_or("GC was not built during eval; young-dict survival checks would be vacuous")?;
    if probe.is_null() {
        return Err("stable GC alloc hook returned null for an instance-sized block".to_string());
    }
    unsafe {
        std::ptr::write_bytes(probe, 0, pyre_object::W_INSTANCE_OBJECT_SIZE);
    }
    Ok(())
}

#[test]
fn young_dict_list_elements_survive_minor_collection() {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_harness)
        .expect("spawn worker thread");
    handle
        .join()
        .expect("worker thread panicked")
        .expect("list young dict gc stress program failed");
}
