//! Consolidated GC-stress integration tests (formerly six separate test
//! binaries: `instance_gc_stress`, `list_elements_gc_stress`,
//! `list_young_dict_gc_stress`, `method_cache_gc_stress`,
//! `type_namespace_gc_stress`, `weak_subclasses_gc_stress`). Merging them into
//! one binary avoids six heavy relinks of the `pyre_jit` + interpreter + object
//! stack in CI.
//!
//! Every `#[test]` runs one end-to-end Python program through the shared
//! `run_harness`, on its own freshly spawned 256 MiB worker thread. The
//! per-worker thread keeps stack-heavy interpreter / tracer recursion off the
//! default test stack. These tests also share process-global GC and builtin
//! type state, so `run_on_worker` serializes them and `run_harness` installs a
//! fresh GC heap for each worker.
//!
//! The harness mirrors the `pyrex` launcher (`pyrex/src/lib.rs` `real_main` +
//! `run_source`) startup shape, then swaps in a per-worker fresh GC before the
//! program allocates. Each program's module body and functions use `while`
//! loops (no `FOR_ITER`), so eval reaches a JIT-eligible frame. `gc.collect()`
//! then forces deterministic collections through the collect hook
//! (`interp_gc.py:7-26 collect`).
//!
//! Non-vacuity is asserted AFTER eval: the stable instance allocator hook
//! must be live, proving the program's objects routed through the real managed
//! heap rather than the leaking `lltype::malloc` Box fallback (which would make
//! the survival checks meaningless).

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks, reset_gc_fresh_for_test};

static GC_STRESS_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Shared harness body for every GC-stress program. Compiles and runs `program`
/// (using `name` as its `sys.argv[0]` / filename) exactly as the `pyrex`
/// launcher would, after installing a fresh per-worker GC heap. An uncaught
/// `assert` in the program surfaces here as `Err`, so a successful return
/// means every read-back assertion held. `vacuity_label` names the survival
/// checks for the post-eval non-vacuity error.
fn run_harness(program: &str, name: &str, vacuity_label: &str) -> Result<(), String> {
    // Mirror `pyrex::real_main` startup, then `pyrex::run_source`.
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    init_jit_hooks();
    // Per-worker fresh GC heap: these tests share the process-global GC
    // singleton, so hide any prior test's residue before this one allocates.
    reset_gc_fresh_for_test();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd);
    importing::set_sys_argv(&[name.to_string()]);

    let code = compile_source_with_filename(program, Mode::Exec, name)
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
    // successful return means every read-back assertion held.
    eval_with_jit(&mut frame).map_err(|e| format!("execution error: {}", e.message))?;

    // Non-vacuity: the stable instance allocator hook is installed by the
    // `JIT_DRIVER` initializer (`driver_pair` -> `set_gc_allocator`). If it is
    // live now, allocations routed through the managed heap rather than the
    // leaking Box fallback, so the survival checks were meaningful.
    let probe = pyre_object::try_gc_alloc_stable(
        pyre_object::W_OBJECT_OBJECT_GC_TYPE_ID,
        pyre_object::W_OBJECT_OBJECT_SIZE,
    )
    .ok_or_else(|| {
        format!("stable GC alloc hook was not live; {vacuity_label} would be vacuous")
    })?;
    if probe.is_null() {
        return Err("stable GC alloc hook returned null for an instance-sized block".to_string());
    }
    // The probe block is never rooted; zero it so any later sweep reads a
    // well-formed (null map/storage) payload before reclaiming it.
    unsafe {
        std::ptr::write_bytes(probe, 0, pyre_object::W_OBJECT_OBJECT_SIZE);
    }
    Ok(())
}

/// Run `run_harness` on its own freshly spawned 256 MiB worker thread and
/// unwrap the result. Mirrors the launcher's worker stack so deep tracer /
/// interpreter recursion does not overflow the default test stack. The tests
/// mutate process-global GC and builtin type state, so each worker run is
/// serialized even when cargo schedules tests in parallel.
fn run_on_worker(
    program: &'static str,
    name: &'static str,
    vacuity_label: &'static str,
    fail_msg: &'static str,
) {
    // These tests share the process-global GC singleton and process-global
    // builtin type state (`object.weak_subclasses`, version tags), which are
    // not thread-safe, so run them one at a time regardless of cargo's
    // parallel test scheduling. Poison-tolerant: one worker panicking must not
    // wedge the rest.
    let _serial = GC_STRESS_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run_harness(program, name, vacuity_label))
        .expect("spawn worker thread");
    handle
        .join()
        .expect("worker thread panicked")
        .expect(fail_msg);
}

/// `W_ObjectObject` is GC-managed: an instance's movable attribute values
/// (list / str / dict, reachable only through mapdict storage slots), a
/// devolved instance's stored values (reached via the storage back-edge), and a
/// materialised `__dict__` view all survive repeated full collections, while
/// dead throwaway instances are reclaimed each round.
///
/// `a` is a non-devolved instance whose list / str / dict attr values live in
/// mapdict storage slots; `b` devolves past the attribute limit, with its
/// `__dict__` materialised while live (`view`) and kept rooted across the
/// collections. Each round allocates fresh garbage and a dead throwaway
/// instance. The returned checksum is reachable only if every live value
/// survived the 100 collections.
#[test]
fn instance_attrs_survive_full_collection() {
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
    run_on_worker(
        PROGRAM,
        "<instance_gc_stress>",
        "instance survival checks",
        "instance gc stress program failed",
    );
}

/// An object-strategy `W_ListObject`'s elements are GC-traced: instances
/// reachable ONLY through a list survive repeated full collections. Regression
/// for the `W_LIST_GC_TYPE_ID` registration that traced `items` as a single
/// non-managed pointer (the `std::alloc`'d `ItemsBlock`) and never reached the
/// elements — a major collection then swept a list element reachable only via
/// the list. Fixed by giving `W_ListObject` a custom trace
/// (`list_object_custom_trace`) that walks the off-GC block, mirroring
/// `W_TupleObject` / `W_SetObject`.
///
/// `objs` (a list literal -> Object strategy) and `grown` (built by `append`,
/// exercising object-strategy growth) hold `Node` instances reachable ONLY
/// through their list. The returned checksum is reachable only if every element
/// survived the 100 collections.
#[test]
fn list_elements_survive_full_collection() {
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
    run_on_worker(
        PROGRAM,
        "<list_elements_gc_stress>",
        "list element survival checks",
        "list elements gc stress program failed",
    );
}

/// Storing a *young* (nursery) value into an object-strategy `W_ListObject`
/// keeps it alive across a minor GC. The list body is old-gen
/// (`try_gc_alloc_stable`); its elements live in an off-GC `ItemsBlock` reached
/// only via `list_object_custom_trace`. A minor collection forwards an old-gen
/// container's young refs ONLY when the container sits in the remembered set,
/// populated exclusively by the write barrier (`try_gc_write_barrier`).
/// Regression for the missing list write barriers in `w_list_append` /
/// `w_list_setitem` / `w_list_insert` / `w_list_setslice` and
/// `w_list_new_with_strategy` creation. Proven non-vacuous: without the
/// barriers the checksum is corrupted / the process faults; with them it is
/// exact.
///
/// Each `{}` is a nursery `W_DICT` reachable only through its list. The four
/// list mutators that store a ref — append, literal creation, setitem, insert —
/// must each barrier the old-gen list so the next minor GC forwards the young
/// dict. The checksum is recoverable only if every stored dict survived the 200
/// collections.
#[test]
fn young_dict_list_elements_survive_minor_collection() {
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
    run_on_worker(
        PROGRAM,
        "<list_young_dict_gc_stress>",
        "young-dict survival checks",
        "list young dict gc stress program failed",
    );
}

/// Interpreter method cache (`baseobjspace::MethodCache`,
/// `typeobject.py:516-552`) correctness under heavy collection: repeated
/// interpreter-mode method lookups, interleaved with full collections, keep
/// returning the correct functions — proving `version_tag` invalidation and the
/// cached results stay consistent and uncorrupted across `gc.collect()`s. The
/// cache is `not we_are_jitted()`-gated, so `run()`'s early loop iterations run
/// in the interpreter (before the trace threshold) and fill / hit the cache
/// between collections. This guards cache *correctness*, not rooting in
/// isolation.
///
/// `C` is a user heap type with three methods. `run()`'s early (interpreter)
/// iterations cache the `m0` lookup and call it across collections; `tail()`
/// then does interpreter-mode hits on all three methods after the GC is warm. A
/// stale cached function pointer would SIGSEGV on the call, or return a wrong
/// value the assertions catch.
#[test]
fn method_cache_stays_correct_across_collections() {
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
    run_on_worker(
        PROGRAM,
        "<method_cache_gc_stress>",
        "method-cache survival check",
        "method cache gc stress program failed",
    );
}

/// A heap type's namespace is GC-rooted: a user class's method (a function), its
/// class attribute (a movable list), and the per-type `__dict__` getset
/// descriptor (whose `fget` is a collectable function) all survive repeated
/// full collections, even when first reached *fresh* after the collections.
/// Heap type objects (`w_type_new`) are Box-immortal, so the collector never
/// fires their `W_TYPE_GC_TYPE_ID` custom trace and never reaches the movable
/// values bound in the type's namespace `DictStorage`. Before the
/// `HEAP_TYPE_REGISTRY` / `walk_type_dicts_gc` / `walk_raw_getset_roots` root
/// walk, this program SIGSEGV'd.
///
/// `C`'s namespace dict holds a method (`method`), a class attribute
/// (`KLASS_ATTR`), and — once `c.__dict__` is first read — the copied
/// `__dict__` getset descriptor. The returned checksum is reachable only if
/// every namespace value (and the descriptor's getter) survived.
#[test]
fn type_namespace_survives_full_collection() {
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
    run_on_worker(
        PROGRAM,
        "<type_namespace_gc_stress>",
        "namespace survival checks",
        "type namespace gc stress program failed",
    );
}

/// A heap type's `weak_subclasses` list is GC-rooted: a subclass recorded in its
/// base's `weak_subclasses` survives repeated full collections so the base's
/// `mutated()` walk still reaches it. `w_type_add_subclass` stores
/// `w_weakref_new(subclass)` — a `try_gc_alloc` young WEAKREF GcStruct — in the
/// base's off-GC `weak_subclasses`. Before this fix `walk_type_dicts_gc`
/// forwarded `bases` and namespace values but NOT `weak_subclasses`, so the
/// first collection reclaimed the WEAKREF and `type.__setattr__`'s `mutated()`
/// walk ran over the freed slot (a UAF that dropped cache invalidation).
///
/// Observable: `mutated()` resets each live subclass's cached
/// `compares_by_identity_status` to UNKNOWN. The program caches `B`'s status as
/// YES (using `B` instances as identity dict keys), churns collections, then
/// defines `A.__eq__` / `A.__hash__`. If `B`'s weakref survived, its status is
/// reset and fresh `B` keys recompute to "compare by value" -> a 2-key dict
/// collapses to 1 entry. A dangling weakref leaves the stale YES (or crashes on
/// deref). `B` is defined *inside* `run()` after the GC is live so its
/// `weak_subclasses` weakref is a real young GC object.
#[test]
fn weak_subclasses_survive_full_collection() {
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
    run_on_worker(
        PROGRAM,
        "<weak_subclasses_gc_stress>",
        "weak_subclasses survival check",
        "weak_subclasses gc stress program failed",
    );
}
