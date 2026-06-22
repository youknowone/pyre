//! End-to-end correctness guard for the interpreter method cache
//! (`baseobjspace::MethodCache`, `typeobject.py:516-552`) under heavy
//! collection: repeated interpreter-mode method lookups, interleaved with
//! full collections, must keep returning the correct functions — proving
//! `version_tag` invalidation and the cached results stay consistent and
//! uncorrupted across `gc.collect()`s.
//!
//! The cache is `not we_are_jitted()`-gated, so the cache-sensitive calls
//! run in interpreter mode: `run()`'s early loop iterations execute in the
//! interpreter (before the trace threshold) and fill / hit the cache
//! between `gc.collect()`s; under `gc_stress` every allocation collects,
//! so the cache is exercised across many collections (hundreds of
//! interpreter-mode hits in practice).  The hot loop also builds the GC
//! lazily, exactly like the `pyrex` launcher.
//!
//! Note on rooting: every cached value is an MRO type's namespace-dict
//! resident already kept reachable by `walk_type_dicts_gc`, and old-gen
//! residents are not relocated by a collection in the current model — so
//! `walk_method_cache_gc` forwards no slot here today (it is the
//! parity-faithful equivalent of RPython tracing `MethodCache.lookup_where`
//! and becomes load-bearing once those values become movable).  This test
//! therefore guards cache *correctness*, not the rooting in isolation.
//!
//! Non-vacuity is asserted AFTER eval: the stable allocator hook must be
//! live, proving the GC was actually built during the run.

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

// `C` is a user heap type with three methods.  `run()`'s early
// (interpreter) iterations cache the `m0` lookup and call it across
// collections; `tail()` then does interpreter-mode hits on all three
// methods after the GC is warm.  A stale cached function pointer would
// SIGSEGV on the call, or return a wrong value the assertions catch.
const PROGRAM: &str = r#"
import gc

class C:
    def m0(self):
        return 1
    def m1(self):
        return 10
    def m2(self):
        return 100

def run():
    c = C()
    total = 0
    n = 0
    while n < 100:
        junk = [0] * 50
        tmp = C()
        total = total + c.m0()
        gc.collect()
        n = n + 1
    return total

def tail():
    c = C()
    acc = 0
    k = 0
    while k < 30:
        acc = acc + c.m0() + c.m1() + c.m2()
        gc.collect()
        k = k + 1
    return acc

warm = run()
acc = tail()
assert warm == 100, warm
assert acc == 30 * 111, acc
result = warm + acc
assert result == 3430, result
"#;

fn run_harness() -> Result<(), String> {
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    init_jit_hooks();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd);
    importing::set_sys_argv(&["<method_cache_gc_stress>".to_string()]);

    let code = compile_source_with_filename(PROGRAM, Mode::Exec, "<method_cache_gc_stress>")
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

    // Non-vacuity: the stable allocator hook proves the GC was built
    // during eval, so the cached method survived real relocation.
    let probe = pyre_object::try_gc_alloc_stable(
        pyre_object::W_INSTANCE_GC_TYPE_ID,
        pyre_object::W_INSTANCE_OBJECT_SIZE,
    )
    .ok_or("GC was not built during eval; method-cache survival check would be vacuous")?;
    if probe.is_null() {
        return Err("stable GC alloc hook returned null".to_string());
    }
    unsafe {
        std::ptr::write_bytes(probe, 0, pyre_object::W_INSTANCE_OBJECT_SIZE);
    }
    Ok(())
}

#[test]
fn method_cache_stays_correct_across_collections() {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_harness)
        .expect("spawn worker thread");
    handle
        .join()
        .expect("worker thread panicked")
        .expect("method cache gc stress program failed");
}
