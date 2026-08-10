//! gc module — PyPy: `pypy/module/gc/`.
//!
//! Partial port of `interp_gc.py`. Explicit collection runs the complete
//! RPython collection, then drains the finalizer queue synchronously.

use pyre_object::*;
use rustpython_wtf8::Wtf8;
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

pub mod hook;

/// `interp_gc.py` tracks a process-wide `enabled` flag on the GC
/// frontend; pyre has no generational threshold knob, but
/// `gc.isenabled()` should reflect the most recent `enable`/`disable`
/// call so callers that toggle and re-read the state stay consistent.
static GC_ENABLED: AtomicBool = AtomicBool::new(true);

/// The collection thresholds `gc.get_threshold()` reports.  pyre's collector
/// has no generational allocation counters to drive, so the values are only
/// remembered: `set_threshold` stores what it was given and `get_threshold`
/// hands the same tuple back, which is the part of the pair's behaviour a
/// caller can observe.  The third threshold is not among them — an incremental
/// collector has no third generation to size, so it reads back as 0 whatever
/// it was set to.  The initial values are the ones a fresh interpreter starts
/// with.
static GC_THRESHOLD: [AtomicI64; 2] = [AtomicI64::new(2000), AtomicI64::new(10)];

const STATE_SCANNING: u8 = 0;
const STATE_MARKING: u8 = 1;
const STATE_SWEEPING: u8 = 2;
const STATE_FINALIZING: u8 = 3;
const STATE_USERDEL: u8 = 4;

/// `rgc.py:52-60 is_done__states`: a major collection has finished when the
/// step ended in the starting state *and* did not start there. A collector
/// with no work to do reports `(0, 0)`, which is not the end of anything.
fn is_done_states(oldstate: u8, newstate: u8) -> bool {
    oldstate != STATE_SCANNING && newstate == STATE_SCANNING
}

/// `interp_gc.py:91-130 StepCollector.finalizing`. `space.fromcache` owns one
/// instance per object space upstream; pyre has one process-wide object space,
/// so the corresponding state is shared rather than thread-local.
static STEP_FINALIZING: AtomicBool = AtomicBool::new(false);

/// `referents.py:11-15 W_GcRef(W_Root)`: an app-level handle for a raw GC
/// object that is not itself a Python object.  The field is deliberately on
/// the wrapper and participates in normal type tracing; a side table would
/// neither keep the referent alive nor receive forwarding updates.
pub mod gcref {
    use super::*;

    #[crate::pyre_class("GcRef")]
    pub struct W_GcRef {
        pub gcref: PyObjectRef,
    }

    #[crate::pyre_methods]
    impl W_GcRef {
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            _args: &[PyObjectRef],
        ) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::PyError::type_error("GcRef() takes no arguments"))
        }
    }

    /// Allocate from a raw target already published at `target_slot` on the
    /// shadow stack.  `type_object()` may allocate while it initializes the
    /// TypeDef, so the target is re-read afterwards.
    pub fn wrap_rooted(target_slot: usize) -> PyObjectRef {
        let w_type = type_object();
        let target = pyre_object::gc_roots::shadow_stack_get(target_slot);
        let value = W_GcRef {
            ob: PyObject {
                ob_type: &GCREF_TYPE,
                w_class: w_type,
            },
            gcref: target,
        };
        pyre_object::lltype::malloc_typed_managed(value) as PyObjectRef
    }

    pub fn unwrap(w_obj: PyObjectRef) -> majit_ir::GcRef {
        W_GcRef::from_obj(w_obj)
            .map(|wrapper| majit_ir::GcRef(wrapper.gcref as usize))
            .unwrap_or(majit_ir::GcRef(w_obj as usize))
    }
}

/// `referents.py:190-241 W_GcStats`.  These are native integer fields on the
/// interpreter object, matching the upstream W_Root rather than a Python dict
/// or a process-global side table.
pub mod stats {
    use super::*;

    #[crate::pyre_class("GcStats")]
    pub struct W_GcStats {
        pub(super) total_memory_pressure: i64,
        pub(super) total_gc_memory: i64,
        pub(super) total_allocated_memory: i64,
        pub(super) peak_memory: i64,
        pub(super) peak_allocated_memory: i64,
        pub(super) jit_backend_allocated: i64,
        pub(super) jit_backend_used: i64,
        pub(super) total_arena_memory: i64,
        pub(super) total_rawmalloced_memory: i64,
        pub(super) peak_arena_memory: i64,
        pub(super) peak_rawmalloced_memory: i64,
        pub(super) nursery_size: i64,
        pub(super) total_gc_time: i64,
    }

    #[crate::pyre_methods]
    impl W_GcStats {
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            _args: &[PyObjectRef],
        ) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::PyError::type_error(
                "object.__new__(GcStats) is not safe, use GcStats.__new__()",
            ))
        }

        #[getter]
        fn total_memory_pressure(&self) -> i64 {
            self.total_memory_pressure
        }
        #[getter]
        fn total_gc_memory(&self) -> i64 {
            self.total_gc_memory
        }
        #[getter]
        fn total_allocated_memory(&self) -> i64 {
            self.total_allocated_memory
        }
        #[getter]
        fn peak_memory(&self) -> i64 {
            self.peak_memory
        }
        #[getter]
        fn peak_allocated_memory(&self) -> i64 {
            self.peak_allocated_memory
        }
        #[getter]
        fn jit_backend_allocated(&self) -> i64 {
            self.jit_backend_allocated
        }
        #[getter]
        fn jit_backend_used(&self) -> i64 {
            self.jit_backend_used
        }
        #[getter]
        fn total_arena_memory(&self) -> i64 {
            self.total_arena_memory
        }
        #[getter]
        fn total_rawmalloced_memory(&self) -> i64 {
            self.total_rawmalloced_memory
        }
        #[getter]
        fn peak_arena_memory(&self) -> i64 {
            self.peak_arena_memory
        }
        #[getter]
        fn peak_rawmalloced_memory(&self) -> i64 {
            self.peak_rawmalloced_memory
        }
        #[getter]
        fn nursery_size(&self) -> i64 {
            self.nursery_size
        }
        #[getter]
        fn total_gc_time(&self) -> i64 {
            self.total_gc_time
        }
    }

    pub fn new(memory_pressure: bool) -> PyObjectRef {
        // `#[pyre_class]::allocate` stamps `get_instantiate(PYTYPE)` into the
        // header. Initialize the TypeDef first so that slot is the real class,
        // not the macro static's pre-init name placeholder.
        let _ = type_object();
        let stats = majit_gc::active_gc_memory_stats();
        let (jit_backend_allocated, jit_backend_used) = majit_gc::active_jit_backend_memory_stats();
        // referents.py:192-195: the optional selector performs the collector's
        // root-reachable `inspector.count_memory_pressure` walk; otherwise it
        // preserves the public -1 sentinel.
        let total_memory_pressure = if memory_pressure {
            majit_gc::total_memory_pressure() as i64
        } else {
            -1
        };
        W_GcStats::allocate(W_GcStats {
            ob: PyObject::default(),
            total_memory_pressure,
            total_gc_memory: stats.total_gc_memory as i64,
            total_allocated_memory: stats.total_allocated_memory as i64,
            peak_memory: stats.peak_memory as i64,
            peak_allocated_memory: stats.peak_allocated_memory as i64,
            jit_backend_allocated: jit_backend_allocated as i64,
            jit_backend_used: jit_backend_used as i64,
            total_arena_memory: stats.total_arena_memory as i64,
            total_rawmalloced_memory: stats.total_rawmalloced_memory as i64,
            peak_arena_memory: stats.peak_arena_memory as i64,
            peak_rawmalloced_memory: stats.peak_rawmalloced_memory as i64,
            nursery_size: stats.nursery_size as i64,
            total_gc_time: stats.total_gc_time_ms as i64,
        })
    }
}

fn collect_step_stat_value(
    args: &[PyObjectRef],
    name: &'static str,
) -> Result<PyObjectRef, crate::PyError> {
    let value = unsafe {
        crate::objspace::std::mapdict::instance_node_getdictvalue(args[1], Wtf8::new(name))
    };
    value.ok_or_else(|| crate::PyError::attribute_error("uninitialized GcCollectStepStats"))
}

macro_rules! collect_step_stat_getter {
    ($function:ident, $name:literal) => {
        fn $function(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            collect_step_stat_value(args, $name)
        }
    };
}

collect_step_stat_getter!(collect_step_count, "_count");
collect_step_stat_getter!(collect_step_duration, "_duration");
collect_step_stat_getter!(collect_step_duration_min, "_duration_min");
collect_step_stat_getter!(collect_step_duration_max, "_duration_max");
collect_step_stat_getter!(collect_step_oldstate, "_oldstate");
collect_step_stat_getter!(collect_step_newstate, "_newstate");
collect_step_stat_getter!(collect_step_major_is_done, "_major_is_done");

fn collect_step_stats_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = crate::baseobjspace::text_w(args[1])?;
    // Same rule as `stats_getattribute`: both types keep their values in
    // mapdict slots under leading-underscore names, so hiding the whole prefix
    // covers a slot added later instead of only the seven that exist today.
    if name == "__dict__" || name.starts_with('_') {
        return Err(crate::PyError::attribute_error(format!(
            "'GcCollectStepStats' object has no attribute '{name}'"
        )));
    }
    crate::baseobjspace::object_getattribute(args[0], name)
}

fn collect_step_stats_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = crate::baseobjspace::text_w(args[1])?;
    Err(crate::PyError::attribute_error(format!(
        "readonly attribute '{name}'"
    )))
}

fn gc_collect_step_stats_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("GcCollectStepStats", |ns| unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                "__getattribute__",
                crate::make_builtin_function_with_arity(
                    "__getattribute__",
                    collect_step_stats_getattribute,
                    2,
                ),
            );
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                "__setattr__",
                crate::make_builtin_function_with_arity(
                    "__setattr__",
                    collect_step_stats_setattr,
                    3,
                ),
            );
            for (name, value) in [
                ("STATE_SCANNING", STATE_SCANNING),
                ("STATE_MARKING", STATE_MARKING),
                ("STATE_SWEEPING", STATE_SWEEPING),
                ("STATE_FINALIZING", STATE_FINALIZING),
                ("STATE_USERDEL", STATE_USERDEL),
            ] {
                pyre_object::w_dict_setitem_str_no_proxy(ns, name, w_int_new(value as i64));
            }
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                "GC_STATES",
                w_tuple_new(
                    ["SCANNING", "MARKING", "SWEEPING", "FINALIZING", "USERDEL"]
                        .into_iter()
                        .map(w_str_new)
                        .collect(),
                ),
            );
            for (name, getter) in [
                ("count", collect_step_count as crate::gateway::BuiltinCodeFn),
                ("duration", collect_step_duration),
                ("duration_min", collect_step_duration_min),
                ("duration_max", collect_step_duration_max),
                ("oldstate", collect_step_oldstate),
                ("newstate", collect_step_newstate),
                ("major_is_done", collect_step_major_is_done),
            ] {
                pyre_object::w_dict_setitem_str_no_proxy(
                    ns,
                    name,
                    crate::typedef::make_getset_descriptor_named(
                        crate::make_builtin_function_with_arity(name, getter, 2),
                        name,
                    ),
                );
            }
        });
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp as usize
    }) as PyObjectRef
}

fn new_collect_step_stats(
    oldstate: u8,
    newstate: u8,
    major_is_done: bool,
) -> Result<PyObjectRef, crate::PyError> {
    new_collect_step_stats_full(1, -1.0, -1.0, -1.0, oldstate, newstate, major_is_done)
}

#[allow(clippy::too_many_arguments)]
fn new_collect_step_stats_full(
    count: i64,
    duration: f64,
    duration_min: f64,
    duration_max: f64,
    oldstate: u8,
    newstate: u8,
    major_is_done: bool,
) -> Result<PyObjectRef, crate::PyError> {
    initialize_stats(
        gc_collect_step_stats_type(),
        &[
            ("_count", StatValue::Int(count)),
            ("_duration", StatValue::Float(duration)),
            ("_duration_min", StatValue::Float(duration_min)),
            ("_duration_max", StatValue::Float(duration_max)),
            ("_oldstate", StatValue::Int(oldstate as i64)),
            ("_newstate", StatValue::Int(newstate as i64)),
            ("_major_is_done", StatValue::Bool(major_is_done)),
        ],
        "GcCollectStepStats",
    )
}

fn readonly_stat_value(
    args: &[PyObjectRef],
    name: &'static str,
    typename: &'static str,
) -> Result<PyObjectRef, crate::PyError> {
    let value = unsafe {
        crate::objspace::std::mapdict::instance_node_getdictvalue(args[1], Wtf8::new(name))
    };
    value.ok_or_else(|| crate::PyError::attribute_error(format!("uninitialized {typename}")))
}

macro_rules! readonly_stat_getter {
    ($function:ident, $field:literal, $typename:literal) => {
        fn $function(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            readonly_stat_value(args, $field, $typename)
        }
    };
}

readonly_stat_getter!(minor_count, "_count", "GcMinorStats");
readonly_stat_getter!(minor_duration, "_duration", "GcMinorStats");
readonly_stat_getter!(minor_duration_min, "_duration_min", "GcMinorStats");
readonly_stat_getter!(minor_duration_max, "_duration_max", "GcMinorStats");
readonly_stat_getter!(
    minor_total_memory_used,
    "_total_memory_used",
    "GcMinorStats"
);
readonly_stat_getter!(minor_pinned_objects, "_pinned_objects", "GcMinorStats");

readonly_stat_getter!(collect_count, "_count", "GcCollectStats");
readonly_stat_getter!(
    collect_num_major_collects,
    "_num_major_collects",
    "GcCollectStats"
);
readonly_stat_getter!(
    collect_arenas_count_before,
    "_arenas_count_before",
    "GcCollectStats"
);
readonly_stat_getter!(
    collect_arenas_count_after,
    "_arenas_count_after",
    "GcCollectStats"
);
readonly_stat_getter!(collect_arenas_bytes, "_arenas_bytes", "GcCollectStats");
readonly_stat_getter!(
    collect_rawmalloc_bytes_before,
    "_rawmalloc_bytes_before",
    "GcCollectStats"
);
readonly_stat_getter!(
    collect_rawmalloc_bytes_after,
    "_rawmalloc_bytes_after",
    "GcCollectStats"
);
readonly_stat_getter!(collect_pinned_objects, "_pinned_objects", "GcCollectStats");

fn stats_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = crate::baseobjspace::text_w(args[1])?;
    Err(crate::PyError::attribute_error(format!(
        "readonly attribute '{name}'"
    )))
}

fn stats_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = crate::baseobjspace::text_w(args[1])?;
    if name == "__dict__" || name.starts_with('_') {
        return Err(crate::PyError::attribute_error(format!(
            "stats object has no attribute '{name}'"
        )));
    }
    crate::baseobjspace::object_getattribute(args[0], name)
}

fn make_private_stats_type(
    name: &'static str,
    fields: &[(&'static str, crate::gateway::BuiltinCodeFn)],
) -> PyObjectRef {
    let tp = crate::typedef::make_builtin_type(name, |ns| unsafe {
        pyre_object::w_dict_setitem_str_no_proxy(
            ns,
            "__getattribute__",
            crate::make_builtin_function_with_arity("__getattribute__", stats_getattribute, 2),
        );
        pyre_object::w_dict_setitem_str_no_proxy(
            ns,
            "__setattr__",
            crate::make_builtin_function_with_arity("__setattr__", stats_setattr, 3),
        );
        for &(field, getter) in fields {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                field,
                crate::typedef::make_getset_descriptor_named(
                    crate::make_builtin_function_with_arity(field, getter, 2),
                    field,
                ),
            );
        }
    });
    unsafe { typeobject::w_type_set_hasdict(tp, true) };
    unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
    tp
}

fn gc_minor_stats_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        make_private_stats_type(
            "GcMinorStats",
            &[
                ("count", minor_count),
                ("duration", minor_duration),
                ("duration_min", minor_duration_min),
                ("duration_max", minor_duration_max),
                ("total_memory_used", minor_total_memory_used),
                ("pinned_objects", minor_pinned_objects),
            ],
        ) as usize
    }) as PyObjectRef
}

fn gc_collect_stats_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        make_private_stats_type(
            "GcCollectStats",
            &[
                ("count", collect_count),
                ("num_major_collects", collect_num_major_collects),
                ("arenas_count_before", collect_arenas_count_before),
                ("arenas_count_after", collect_arenas_count_after),
                ("arenas_bytes", collect_arenas_bytes),
                ("rawmalloc_bytes_before", collect_rawmalloc_bytes_before),
                ("rawmalloc_bytes_after", collect_rawmalloc_bytes_after),
                ("pinned_objects", collect_pinned_objects),
            ],
        ) as usize
    }) as PyObjectRef
}

/// One private stats field, still unbuilt.
///
/// The point of deferring is rooting: a `Vec<(&str, PyObjectRef)>` built up
/// front holds every value in plain Rust memory while the remaining `w_*_new`
/// calls run, and the collector forwards shadow-stack slots, not Rust locals.
enum StatValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl StatValue {
    fn materialize(&self) -> PyObjectRef {
        match *self {
            StatValue::Int(v) => w_int_new(v),
            StatValue::Float(v) => w_float_new(v),
            StatValue::Bool(v) => w_bool_from(v),
        }
    }
}

/// Allocate a private stats instance of `stats_type` and fill it.
///
/// Both the field constructors and the mapdict transition inside
/// `instance_node_setdictvalue` allocate, so either can move the instance and
/// the value a Rust local names. Pin each on the shadow stack and read them
/// back through their slots for every store, the way `populate_public_gc_stats`
/// below does. The instance is created here rather than passed in so a caller
/// cannot hand over one that was allocated before any root existed.
fn initialize_stats(
    stats_type: PyObjectRef,
    fields: &[(&'static str, StatValue)],
    typename: &'static str,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let stats_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_instance_new(stats_type));
    for (name, value) in fields {
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(value.materialize());
        let stored = unsafe {
            crate::objspace::std::mapdict::instance_node_setdictvalue(
                pyre_object::gc_roots::shadow_stack_get(stats_slot),
                Wtf8::new(name),
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            )
        };
        if !stored {
            return Err(crate::PyError::attribute_error(format!(
                "cannot initialize {typename}"
            )));
        }
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(stats_slot))
}

fn new_minor_stats(
    count: i64,
    duration: f64,
    duration_min: f64,
    duration_max: f64,
    total_memory_used: usize,
    pinned_objects: usize,
) -> Result<PyObjectRef, crate::PyError> {
    initialize_stats(
        gc_minor_stats_type(),
        &[
            ("_count", StatValue::Int(count)),
            ("_duration", StatValue::Float(duration)),
            ("_duration_min", StatValue::Float(duration_min)),
            ("_duration_max", StatValue::Float(duration_max)),
            (
                "_total_memory_used",
                StatValue::Int(total_memory_used as i64),
            ),
            ("_pinned_objects", StatValue::Int(pinned_objects as i64)),
        ],
        "GcMinorStats",
    )
}

#[allow(clippy::too_many_arguments)]
fn new_collect_stats(
    count: i64,
    num_major_collects: usize,
    arenas_count_before: usize,
    arenas_count_after: usize,
    arenas_bytes: usize,
    rawmalloc_bytes_before: usize,
    rawmalloc_bytes_after: usize,
    pinned_objects: usize,
) -> Result<PyObjectRef, crate::PyError> {
    initialize_stats(
        gc_collect_stats_type(),
        &[
            ("_count", StatValue::Int(count)),
            (
                "_num_major_collects",
                StatValue::Int(num_major_collects as i64),
            ),
            (
                "_arenas_count_before",
                StatValue::Int(arenas_count_before as i64),
            ),
            (
                "_arenas_count_after",
                StatValue::Int(arenas_count_after as i64),
            ),
            ("_arenas_bytes", StatValue::Int(arenas_bytes as i64)),
            (
                "_rawmalloc_bytes_before",
                StatValue::Int(rawmalloc_bytes_before as i64),
            ),
            (
                "_rawmalloc_bytes_after",
                StatValue::Int(rawmalloc_bytes_after as i64),
            ),
            ("_pinned_objects", StatValue::Int(pinned_objects as i64)),
        ],
        "GcCollectStats",
    )
}

fn pin_object(object: majit_ir::GcRef) {
    pyre_object::gc_roots::pin_root(object.0 as PyObjectRef);
}

/// `referents.py:53-78 _list_w_obj_referents`: push the app-level objects
/// `w_obj` refers to directly onto the shadow stack. The walk looks through
/// the interpreter-internal structs in between, so a list reports its items
/// and not the array holding them.
///
/// Only managed-heap referents are reported, the same boundary `gc.get_objects`
/// and `gc.is_tracked` draw. An immortal referent carries a GC header but sits
/// outside the collector's ranges, and a slot such as `ob_type` can hold a
/// static that has no header at all, so there is no address the walk could
/// safely widen to.
fn pin_referents(w_obj: PyObjectRef) {
    majit_gc::get_referents(majit_ir::GcRef(w_obj as usize), pin_object);
}

/// Wrap every raw collector node rooted in `[first, last)` as
/// `referents.py:35-39 wrap`: app-level objects pass through, internal nodes
/// become `W_GcRef`.  Results are rooted as they are made because constructing
/// a later wrapper can initialize a type and allocate.
/// Wrap the raw nodes rooted at `first..last`, leaving the wrappers pinned.
///
/// Returns the first shadow-stack slot of the wrapped range, which runs to the
/// stack top on return. A slot range rather than a `Vec<PyObjectRef>` because
/// every caller allocates a list next, and only the slots are forwarded.
fn wrap_raw_nodes(first: usize, last: usize) -> usize {
    let result_first = pyre_object::gc_roots::shadow_stack_len();
    for slot in first..last {
        let raw = majit_ir::GcRef(pyre_object::gc_roots::shadow_stack_get(slot) as usize);
        let wrapped = if majit_gc::is_app_level_object(raw) {
            // The query can park behind a moving collection; reload the slot.
            pyre_object::gc_roots::shadow_stack_get(slot)
        } else {
            gcref::wrap_rooted(slot)
        };
        pyre_object::gc_roots::pin_root(wrapped);
    }
    result_first
}

/// Build a list holding the objects rooted at `slots`.
///
/// The list is allocated before any slot is read, and both the list and the
/// element are re-read around every append. Gathering the elements into a
/// `Vec<PyObjectRef>` first and allocating afterwards would hand the list
/// pre-copy addresses as soon as one of these allocations collects: the
/// collector forwards shadow-stack entries, not Rust vectors. Being pinned
/// keeps an object alive, which is not the same as keeping a copy of its
/// address valid.
fn list_from_root_slots(slots: impl IntoIterator<Item = usize>) -> PyObjectRef {
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_list_new_empty());
    for slot in slots {
        unsafe {
            w_list_append(
                pyre_object::gc_roots::shadow_stack_get(list_slot),
                pyre_object::gc_roots::shadow_stack_get(slot),
            );
        }
    }
    pyre_object::gc_roots::shadow_stack_get(list_slot)
}

/// `list_from_root_slots` over every slot pinned since `first`.
fn list_from_roots(first: usize) -> PyObjectRef {
    list_from_root_slots(first..pyre_object::gc_roots::shadow_stack_len())
}

fn user_del_action() -> Option<&'static mut crate::executioncontext::UserDelAction> {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return None;
    }
    let action = unsafe { (*ec).user_del_action };
    if action.is_null() {
        None
    } else {
        Some(unsafe { &mut *action })
    }
}

fn enable_finalizers(action: &mut crate::executioncontext::UserDelAction) {
    if action.finalizers_lock_count == 0 {
        return;
    }
    action.finalizers_lock_count -= 1;
    if action.finalizers_lock_count == 0 {
        if let Some(pending) = action.pending_with_disabled_del.take() {
            // The list just left its GC-visible UserDelAction slot; keep every
            // entry rooted while the finalizers run (upstream clears the
            // GC-visible list as it progresses, interp_gc.py:80-84).
            let _roots = pyre_object::gc_roots::push_roots();
            for &obj in pending.iter() {
                pyre_object::gc_roots::pin_root(obj);
            }
            let root_end = pyre_object::gc_roots::shadow_stack_len();
            let root_base = root_end - pending.len();
            for index in 0..pending.len() {
                action._call_finalizer(pyre_object::gc_roots::shadow_stack_get(root_base + index));
            }
        }
    }
}

fn disable_finalizers(action: &mut crate::executioncontext::UserDelAction) {
    action.finalizers_lock_count += 1;
    if action.pending_with_disabled_del.is_none() {
        action.pending_with_disabled_del = Some(Vec::new());
    }
}

fn run_finalizers_now() {
    if let Some(action) = user_del_action() {
        let temp_reenable = !action.enabled_at_app_level;
        if temp_reenable {
            enable_finalizers(action);
        }
        action._run_finalizers();
        if temp_reenable {
            disable_finalizers(action);
        }
    }
}

fn format_gc_stat(value: i64) -> String {
    if value < 1_000_000 {
        format!("{:.1}kB", value as f64 / 1024.0)
    } else {
        format!("{:.1}MB", value as f64 / 1024.0 / 1024.0)
    }
}

fn gc_stats_public_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("GcStats", |ns| unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                "__init__",
                crate::make_builtin_function_with_arity("__init__", gc_stats_public_init, 2),
            );
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                "__repr__",
                crate::make_builtin_function_with_arity("__repr__", gc_stats_repr, 1),
            );
            pyre_object::w_dict_setitem_str_no_proxy(ns, "__dict__", crate::typedef::dict_descr());
        });
        // app_referents.py:GcStats is an ordinary app-level class.
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

fn gc_stats_public_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    populate_public_gc_stats(args[0], args[1])?;
    Ok(w_none())
}

fn gc_stats_attr_string(self_slot: usize, name: &'static str) -> Result<String, crate::PyError> {
    let value =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(self_slot), name)?;
    Ok(unsafe { crate::display::py_str_wtf8(value)? }
        .to_string_lossy()
        .into_owned())
}

fn gc_stats_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(args[0]);
    let raw =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(self_slot), "_s")?;
    pyre_object::gc_roots::pin_root(raw);
    let raw_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let raw = stats::W_GcStats::from_obj(pyre_object::gc_roots::shadow_stack_get(raw_slot))
        .ok_or_else(|| crate::PyError::type_error("GcStats._s is not a native GcStats"))?;
    let total_memory_pressure = raw.total_memory_pressure;
    let total_arena_allocated = format_gc_stat(
        raw.total_allocated_memory - raw.total_rawmalloced_memory - raw.nursery_size,
    );

    // app_referents.py:76-120. Read the public attributes afresh: this class
    // has a normal instance dict upstream, and user mutation affects repr.
    let total_gc_memory = gc_stats_attr_string(self_slot, "total_gc_memory")?;
    let peak_memory = gc_stats_attr_string(self_slot, "peak_memory")?;
    let total_arena_memory = gc_stats_attr_string(self_slot, "total_arena_memory")?;
    let peak_arena_memory = gc_stats_attr_string(self_slot, "peak_arena_memory")?;
    let total_rawmalloced_memory = gc_stats_attr_string(self_slot, "total_rawmalloced_memory")?;
    let peak_rawmalloced_memory = gc_stats_attr_string(self_slot, "peak_rawmalloced_memory")?;
    let nursery_size = gc_stats_attr_string(self_slot, "nursery_size")?;
    let jit_backend_used = gc_stats_attr_string(self_slot, "jit_backend_used")?;
    let total_memory_pressure_text = gc_stats_attr_string(self_slot, "total_memory_pressure")?;
    let memory_used_sum = gc_stats_attr_string(self_slot, "memory_used_sum")?;
    let total_allocated_memory = gc_stats_attr_string(self_slot, "total_allocated_memory")?;
    let peak_allocated_memory = gc_stats_attr_string(self_slot, "peak_allocated_memory")?;
    let jit_backend_allocated = gc_stats_attr_string(self_slot, "jit_backend_allocated")?;
    let memory_allocated_sum = gc_stats_attr_string(self_slot, "memory_allocated_sum")?;
    let total_gc_time_obj = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "total_gc_time",
    )?;
    pyre_object::gc_roots::pin_root(total_gc_time_obj);
    let total_gc_time_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let total_gc_time =
        crate::baseobjspace::float_w(pyre_object::gc_roots::shadow_stack_get(total_gc_time_slot))?
            / 1000.0;
    let extra = if total_memory_pressure != -1 {
        format!("\n    memory pressure:         {total_memory_pressure_text}")
    } else {
        String::new()
    };
    Ok(w_str_new(&format!(
        concat!(
            "Total memory consumed:\n",
            "    GC used:                 {total_gc_memory} (peak: {peak_memory})\n",
            "       in arenas:            {total_arena_memory} (peak: {peak_arena_memory})\n",
            "       rawmalloced:          {total_rawmalloced_memory} (peak: {peak_rawmalloced_memory})\n",
            "       nursery:              {nursery_size}\n",
            "    raw assembler used:      {jit_backend_used}{extra}\n",
            "    -----------------------------\n",
            "    Total:                   {memory_used_sum}\n\n",
            "    Total memory allocated (includes freelists):\n",
            "    GC allocated:            {total_allocated_memory} (peak: {peak_allocated_memory})\n",
            "       in arenas:            {total_arena_allocated}\n",
            "       rawmalloced:          {total_rawmalloced_memory}\n",
            "       nursery:              {nursery_size}\n",
            "    raw assembler allocated: {jit_backend_allocated}{extra}\n",
            "    -----------------------------\n",
            "    Total:                   {memory_allocated_sum}\n\n",
            "    Total time spent in GC:  {total_gc_time}\n    "
        ),
        total_gc_memory = total_gc_memory,
        peak_memory = peak_memory,
        total_arena_memory = total_arena_memory,
        peak_arena_memory = peak_arena_memory,
        total_rawmalloced_memory = total_rawmalloced_memory,
        peak_rawmalloced_memory = peak_rawmalloced_memory,
        nursery_size = nursery_size,
        jit_backend_used = jit_backend_used,
        extra = extra,
        memory_used_sum = memory_used_sum,
        total_allocated_memory = total_allocated_memory,
        peak_allocated_memory = peak_allocated_memory,
        total_arena_allocated = total_arena_allocated,
        jit_backend_allocated = jit_backend_allocated,
        memory_allocated_sum = memory_allocated_sum,
        total_gc_time = total_gc_time,
    )))
}

fn populate_public_gc_stats(obj: PyObjectRef, raw: PyObjectRef) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(obj);
    let raw_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(raw);
    let raw = stats::W_GcStats::from_obj(pyre_object::gc_roots::shadow_stack_get(raw_slot))
        .ok_or_else(|| crate::PyError::type_error("GcStats() requires a native GcStats"))?;
    let memory_pressure_value = if raw.total_memory_pressure == -1 {
        0
    } else {
        raw.total_memory_pressure
    };
    let formatted = [
        ("total_gc_memory", raw.total_gc_memory),
        ("jit_backend_used", raw.jit_backend_used),
        ("total_memory_pressure", raw.total_memory_pressure),
        ("total_allocated_memory", raw.total_allocated_memory),
        ("jit_backend_allocated", raw.jit_backend_allocated),
        ("peak_memory", raw.peak_memory),
        ("peak_allocated_memory", raw.peak_allocated_memory),
        ("total_arena_memory", raw.total_arena_memory),
        ("total_rawmalloced_memory", raw.total_rawmalloced_memory),
        ("nursery_size", raw.nursery_size),
        ("peak_arena_memory", raw.peak_arena_memory),
        ("peak_rawmalloced_memory", raw.peak_rawmalloced_memory),
        (
            "memory_used_sum",
            raw.total_gc_memory + memory_pressure_value + raw.jit_backend_used,
        ),
        (
            "memory_allocated_sum",
            raw.total_allocated_memory + memory_pressure_value + raw.jit_backend_allocated,
        ),
    ];
    let total_gc_time = raw.total_gc_time;
    crate::baseobjspace::setattr_str(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        "_s",
        pyre_object::gc_roots::shadow_stack_get(raw_slot),
    )?;
    for (name, value) in formatted {
        let text = w_str_new(&format_gc_stat(value));
        pyre_object::gc_roots::pin_root(text);
        let text_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        crate::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            name,
            pyre_object::gc_roots::shadow_stack_get(text_slot),
        )?;
    }
    // Build and pin the value before reading `obj_slot`: Rust evaluates the
    // receiver first, so an inline `w_int_new` here would allocate — and
    // possibly collect — after the argument already held a raw address.
    let time_value = w_int_new(total_gc_time);
    pyre_object::gc_roots::pin_root(time_value);
    let time_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    crate::baseobjspace::setattr_str(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        "total_gc_time",
        pyre_object::gc_roots::shadow_stack_get(time_slot),
    )?;
    Ok(())
}

fn new_public_gc_stats(memory_pressure: bool) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let raw = stats::new(memory_pressure);
    let raw_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(raw);
    let public_type = gc_stats_public_type();
    let obj = w_instance_new(public_type);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(obj);
    populate_public_gc_stats(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        pyre_object::gc_roots::shadow_stack_get(raw_slot),
    )?;
    Ok(pyre_object::gc_roots::shadow_stack_get(obj_slot))
}

fn gc_call_method(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let result = crate::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::runtime_error("method call failed")))
    } else {
        Ok(result)
    }
}

fn dump_rpy_heap_fd(fd: i32) -> Result<(), crate::PyError> {
    match majit_gc::dump_rpy_heap(fd) {
        Ok(true) => Ok(()),
        Ok(false) => Err(crate::PyError::not_implemented(
            "operation not implemented by this GC",
        )),
        Err(errno) => Err(crate::PyError::os_error_with_errno(
            errno,
            "raw_os_write failed",
        )),
    }
}

fn typeids_z_bytes() -> Result<Vec<u8>, crate::PyError> {
    let text = majit_gc::get_typeids_text()
        .ok_or_else(|| crate::PyError::not_implemented("operation not implemented by this GC"))?;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    encoder
        .write_all(&text)
        .and_then(|_| encoder.finish())
        .map_err(|error| crate::PyError::os_error(error.to_string()))
}

/// The spelling each typeid sidecar wants. `app_referents.py:34` opens
/// `typeids.txt` binary and writes the decompressed bytes; `:40` opens
/// `typeids.lst` text and writes a str, so the object handed to `write`
/// differs even though the surrounding steps do not.
enum TypeidsPayload<'a> {
    Binary(&'a [u8]),
    Text(&'a str),
}

/// `app_referents.py:32,38` `os.path.exists`. Under sandbox the probe is a
/// controller round trip, the way `importing.rs`'s `SeamSourceProvider` does
/// it; `Path::exists` stats the real filesystem, which is exactly what the
/// jail is there to prevent. It escapes the `disallowed-methods` fence only
/// because that list names `std::fs::metadata` rather than the `Path` method
/// wrapping it.
fn typeids_sidecar_exists(path: &std::path::Path) -> bool {
    #[cfg(feature = "sandbox")]
    {
        use std::os::unix::ffi::OsStrExt;
        crate::host_seam::ops::stat(path.as_os_str().as_bytes()).is_ok()
    }
    #[cfg(not(feature = "sandbox"))]
    {
        path.exists()
    }
}

/// `app_referents.py:33-36,39-42`: open the sidecar, write it once, close it.
///
/// The write goes through `builtin_open` and the file object's own `write` /
/// `close`, which is both what upstream writes and what a sandbox build routes
/// to the controller — `std::fs::write` would reach the real filesystem from
/// inside the jail. Errors are left to propagate from those calls, as they do
/// upstream, so the raised `OSError` carries the sidecar's own name rather
/// than the dump's.
fn write_typeids_sidecar(
    path: &std::path::Path,
    payload: TypeidsPayload<'_>,
) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(crate::gateway::fsdecode_os_str(path.as_os_str()));
    let mode_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_str_new(match payload {
        TypeidsPayload::Binary(_) => "wb",
        TypeidsPayload::Text(_) => "w",
    }));
    let opened = crate::builtins::builtin_open(&[
        pyre_object::gc_roots::shadow_stack_get(name_slot),
        pyre_object::gc_roots::shadow_stack_get(mode_slot),
    ])?;
    let opened_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(opened);
    // Materialize the payload only now: `builtin_open` allocates, so a value
    // boxed before it would have to be rooted across the open for nothing.
    let data_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(match payload {
        TypeidsPayload::Binary(bytes) => w_bytes_from_bytes(bytes),
        TypeidsPayload::Text(text) => w_str_new(text),
    });
    let written = gc_call_method(
        pyre_object::gc_roots::shadow_stack_get(opened_slot),
        "write",
        &[pyre_object::gc_roots::shadow_stack_get(data_slot)],
    );
    let closed = gc_call_method(
        pyre_object::gc_roots::shadow_stack_get(opened_slot),
        "close",
        &[],
    );
    written?;
    closed?;
    Ok(())
}

fn dump_rpy_heap_public(file: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let file_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(file);
    // Read the slot at every use instead of caching it in a local: the
    // collector forwards the shadow-stack entry, not the Rust binding, and
    // `w_str_new`, `builtin_open`, `getattr_str`, and `gc_call_method` below
    // all allocate between the uses.
    let file = || pyre_object::gc_roots::shadow_stack_get(file_slot);

    if unsafe { is_str(file()) } {
        // app_referents.py:22-40: filename arm opens/truncates the binary
        // dump, closes it, then materializes typeids.txt/.lst if absent.
        let path = crate::gateway::fspath_buf(file())?;
        let mode_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_str_new("wb"));
        let opened = crate::builtins::builtin_open(&[
            file(),
            pyre_object::gc_roots::shadow_stack_get(mode_slot),
        ])?;
        pyre_object::gc_roots::pin_root(opened);
        let opened_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let fileno = gc_call_method(
            pyre_object::gc_roots::shadow_stack_get(opened_slot),
            "fileno",
            &[],
        )?;
        let fd = crate::baseobjspace::int_w(fileno)? as i32;
        let dump_result = dump_rpy_heap_fd(fd);
        let close_result = gc_call_method(
            pyre_object::gc_roots::shadow_stack_get(opened_slot),
            "close",
            &[],
        );
        dump_result?;
        close_result?;

        let directory = path.parent().unwrap_or_else(|| std::path::Path::new(""));
        let typeids_txt = directory.join("typeids.txt");
        if !typeids_sidecar_exists(&typeids_txt) {
            let text = majit_gc::get_typeids_text().ok_or_else(|| {
                crate::PyError::not_implemented("operation not implemented by this GC")
            })?;
            write_typeids_sidecar(&typeids_txt, TypeidsPayload::Binary(&text))?;
        }
        let typeids_lst = directory.join("typeids.lst");
        if !typeids_sidecar_exists(&typeids_lst) {
            let list = majit_gc::get_typeids_list().ok_or_else(|| {
                crate::PyError::not_implemented("operation not implemented by this GC")
            })?;
            let data: String = list.into_iter().map(|value| format!("{value}\n")).collect();
            write_typeids_sidecar(&typeids_lst, TypeidsPayload::Text(&data))?;
        }
        return Ok(w_none());
    }

    let fd = if unsafe { is_int(file()) } {
        crate::baseobjspace::int_w(file())? as i32
    } else {
        // app_referents.py:44-49: flush only when the attribute exists, then
        // ask for fileno. AttributeError is the only absence case upstream;
        // a present flush method's exception propagates.
        match crate::baseobjspace::getattr_str(file(), "flush") {
            Ok(flush) => {
                crate::call::call_function_impl_result(flush, &[])?;
            }
            Err(error) if error.kind == crate::PyErrorKind::AttributeError => {}
            Err(error) => return Err(error),
        }
        let fileno = gc_call_method(file(), "fileno", &[])?;
        crate::baseobjspace::int_w(fileno)? as i32
    };
    dump_rpy_heap_fd(fd)?;
    Ok(w_none())
}

crate::py_module! {
    "gc",
    interpleveldefs: {
        "callbacks"           => w_list_new(vec![]),
        "garbage"             => w_list_new(vec![]),
        "DEBUG_STATS"         => w_int_new(1),
        "DEBUG_COLLECTABLE"   => w_int_new(2),
        "DEBUG_UNCOLLECTABLE" => w_int_new(4),
        "DEBUG_SAVEALL"       => w_int_new(32),
        "DEBUG_LEAK"          => w_int_new(38),
        "GcCollectStepStats"  => gc_collect_step_stats_type(),
        "GcRef"               => gcref::type_object(),
        "hooks"               => hook::hooks_object(),
    },
    inline_functions: {
        fn collect(
            #[default(w_int_new(0))] generation: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // PyPy `interp_gc.py:7-26 collect` unwraps the optional generation
            // as an int, but deliberately ignores its value.  In particular,
            // unlike CPython's three-generation frontend, every integer is
            // accepted and the default is 0.
            let _generation = crate::baseobjspace::int_w(
                crate::baseobjspace::space_index(generation)?,
            )?;
            crate::baseobjspace::clear_method_cache();
            crate::objspace::std::mapdict::clear_map_attr_cache();
            pyre_object::gc_hook::try_gc_collect();
            run_finalizers_now();
            Ok(w_none())
        }

        fn collect_step() -> Result<PyObjectRef, crate::PyError> {
            // interp_gc.py:91-130 StepCollector: the app-level finalizer drain
            // is a virtual fifth state after the collector has returned to
            // SCANNING.
            if STEP_FINALIZING.load(Ordering::Acquire) {
                run_finalizers_now();
                STEP_FINALIZING.store(false, Ordering::Release);
                return new_collect_step_stats(STATE_USERDEL, STATE_SCANNING, true);
            }

            let (oldstate, mut newstate) = pyre_object::gc_hook::try_gc_collect_step();
            if is_done_states(oldstate, newstate) {
                newstate = STATE_USERDEL;
                STEP_FINALIZING.store(true, Ordering::Release);
            }
            new_collect_step_stats(oldstate, newstate, false)
        }

        fn get_objects(
            #[default(w_none())] generation: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // PyPy `referents.py:112-123`: the audit event precedes argument
            // validation and always carries -1, even for a rejected argument.
            let _generation_root = pyre_object::gc_roots::push_roots();
            let generation_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(generation);
            crate::module::sys::vm::audit("gc.get_objects", &[w_int_new(-1)])?;
            let generation = pyre_object::gc_roots::shadow_stack_get(generation_slot);
            if !unsafe { is_none(generation) } {
                return Err(crate::PyError::not_implemented(
                    "get_objects(generation=None) accepts only None on PyPy",
                ));
            }
            let _roots = pyre_object::gc_roots::push_roots();
            let first = pyre_object::gc_roots::shadow_stack_len();
            majit_gc::get_objects(-1, pin_object);
            Ok(list_from_roots(first))
        }

        fn _get_stats(
            #[default(w_bool_from(false))] memory_pressure: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // referents.py:240-241 `@unwrap_spec(memory_pressure=bool)`.
            Ok(stats::new(crate::baseobjspace::is_true(memory_pressure)?))
        }

        fn get_stats(
            #[default(w_bool_from(false))] memory_pressure: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            new_public_gc_stats(crate::baseobjspace::is_true(memory_pressure)?)
        }

        fn dump_rpy_heap(file: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            dump_rpy_heap_public(file)
        }
    },
    functions: {
        "disable_finalizers" / 0 = |_| {
            // `interp_gc.py:85-89`: this lock is recursive and deliberately
            // independent of gc.isenabled().
            if let Some(action) = user_del_action() {
                disable_finalizers(action);
            }
            Ok(w_none())
        },
        "enable_finalizers" / 0 = |_| {
            // `interp_gc.py:72-84`: unlike gc.enable(), an unmatched public
            // enable is an error rather than a no-op.
            if let Some(action) = user_del_action() {
                if action.finalizers_lock_count == 0 {
                    return Err(crate::PyError::value_error(
                        "finalizers are already enabled",
                    ));
                }
                enable_finalizers(action);
            }
            Ok(w_none())
        },
        "disable"       / 0 = |_| {
            pyre_object::gc_hook::try_gc_set_enabled(false);
            GC_ENABLED.store(false, Ordering::Relaxed);
            if let Some(action) = user_del_action() {
                if action.enabled_at_app_level {
                    action.enabled_at_app_level = false;
                    disable_finalizers(action);
                }
            }
            Ok(w_none())
        },
        "enable"        / 0 = |_| {
            pyre_object::gc_hook::try_gc_set_enabled(true);
            GC_ENABLED.store(true, Ordering::Relaxed);
            if let Some(action) = user_del_action() {
                if !action.enabled_at_app_level {
                    action.enabled_at_app_level = true;
                    enable_finalizers(action);
                }
            }
            Ok(w_none())
        },
        "isenabled"     / 0 = |_| {
            let enabled = match user_del_action() {
                Some(action) => action.enabled_at_app_level,
                None => GC_ENABLED.load(Ordering::Relaxed),
            };
            Ok(w_bool_from(enabled))
        },
        "get_referrers" / * = |args| {
            // referents.py:147-169 get_referrers: list every app-level object,
            // then keep the ones whose direct referents include an argument.
            //
            // The argument scan at `referents.py:166-168` has no `break`, and
            // the multiplicity that follows from that is the contract: an
            // object referring to the same argument twice is reported once
            // (the membership test collapses it), but one that refers to two
            // of the arguments — or one argument passed twice — is reported
            // once per match.
            let _roots = pyre_object::gc_roots::push_roots();
            let args_base = pyre_object::gc_roots::pin_roots(args);
            let mut rooted_args = vec![std::ptr::null_mut(); args.len()];
            pyre_object::gc_roots::shadow_stack_copy_range(args_base, &mut rooted_args);
            crate::module::sys::vm::audit("gc.get_referrers", &rooted_args)?;
            let all_first = pyre_object::gc_roots::shadow_stack_len();
            majit_gc::get_objects(-1, pin_object);
            let all_last = pyre_object::gc_roots::shadow_stack_len();
            // Accumulate the matches as slot indices, not as addresses: the
            // entries stay pinned in `all_first..all_last`, but a copy of one
            // of their addresses goes stale the moment the list allocation
            // below moves the object it names.
            let mut result = Vec::new();
            for slot in all_first..all_last {
                let w_obj = pyre_object::gc_roots::shadow_stack_get(slot);
                let _refs = pyre_object::gc_roots::push_roots();
                let refs_first = pyre_object::gc_roots::shadow_stack_len();
                pin_referents(w_obj);
                let refs_last = pyre_object::gc_roots::shadow_stack_len();
                for index in 0..args.len() {
                    let w_arg = pyre_object::gc_roots::shadow_stack_get(args_base + index);
                    if (refs_first..refs_last)
                        .any(|s| pyre_object::gc_roots::shadow_stack_get(s) == w_arg)
                    {
                        result.push(slot);
                    }
                }
            }
            Ok(list_from_root_slots(result))
        },
        "get_referents" / * = |args| {
            // referents.py:128-145 get_referents.
            let _roots = pyre_object::gc_roots::push_roots();
            let args_base = pyre_object::gc_roots::pin_roots(args);
            let mut rooted_args = vec![std::ptr::null_mut(); args.len()];
            pyre_object::gc_roots::shadow_stack_copy_range(args_base, &mut rooted_args);
            crate::module::sys::vm::audit("gc.get_referents", &rooted_args)?;
            let first = pyre_object::gc_roots::shadow_stack_len();
            for index in 0..args.len() {
                let w_obj = pyre_object::gc_roots::shadow_stack_get(args_base + index);
                pin_referents(w_obj);
            }
            Ok(list_from_roots(first))
        },
        "get_rpy_roots" / 0 = |_| {
            let _roots = pyre_object::gc_roots::push_roots();
            let first = pyre_object::gc_roots::shadow_stack_len();
            if !majit_gc::get_rpy_roots(pin_object) {
                return Err(crate::PyError::not_implemented(
                    "operation not implemented by this GC",
                ));
            }
            let last = pyre_object::gc_roots::shadow_stack_len();
            Ok(list_from_roots(wrap_raw_nodes(first, last)))
        },
        "get_rpy_referents" / 1 = |args| {
            let _roots = pyre_object::gc_roots::push_roots();
            let obj_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(args[0]);
            let raw = gcref::unwrap(pyre_object::gc_roots::shadow_stack_get(obj_slot));
            let first = pyre_object::gc_roots::shadow_stack_len();
            if !majit_gc::get_rpy_referents(raw, pin_object) {
                return Err(crate::PyError::not_implemented(
                    "operation not implemented by this GC",
                ));
            }
            let last = pyre_object::gc_roots::shadow_stack_len();
            Ok(list_from_roots(wrap_raw_nodes(first, last)))
        },
        // `set_threshold(threshold0, threshold1=None, threshold2=None)` — the
        // optional tail leaves no single natural arity, so the body enforces
        // the count itself.
        "set_threshold" / * = |args| {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            if crate::builtins::has_real_kwargs(kwargs) {
                return Err(crate::PyError::type_error(
                    "set_threshold() takes no keyword arguments",
                ));
            }
            // CPython 3.14 `gc.set_threshold(threshold0[, threshold1[,
            // threshold2]])` writes only the positions it was given, and
            // parses every argument before writing any of them.
            if positional.is_empty() || positional.len() > 3 {
                return Err(crate::PyError::type_error(
                    "gc.set_threshold requires 1 to 3 arguments",
                ));
            }
            // Read every value before storing any, so a non-integer in the
            // tail leaves the previous thresholds untouched.  An omitted
            // trailing value keeps the threshold it already had, and a third
            // value is validated but not kept.
            let mut given = Vec::with_capacity(positional.len());
            for &w_value in positional {
                // The index protocol, so an object carrying only `__int__` is
                // a TypeError rather than a silent conversion.
                given.push(crate::builtins::space_index_w(w_value)?);
            }
            for (slot, value) in GC_THRESHOLD.iter().zip(given) {
                slot.store(value, Ordering::Relaxed);
            }
            Ok(w_none())
        },
        "get_threshold" / 0 = |_| Ok(w_tuple_new(
            GC_THRESHOLD
                .iter()
                .map(|slot| w_int_new(slot.load(Ordering::Relaxed)))
                .chain(std::iter::once(w_int_new(0)))
                .collect(),
        )),
        "get_count"     / 0 = |_| Ok(w_tuple_new(vec![
            w_int_new(0), w_int_new(0), w_int_new(0),
        ])),
        "is_tracked"    / 1 = |args| {
            // CPython 3.14 `gc.is_tracked(obj)`: whether the collector
            // traverses references out of the object. Asked of the registered
            // type rather than of the heap the instance landed in, so an int
            // answers the same under `PYRE_GC_INTERP`, under the JIT and on
            // wasm as it does on the immortal path.
            //
            // `bytes` and `int` answer True where CPython answers False: their
            // pyre structs carry `w_dict` and `w_weakreflifeline` slots that
            // the collector really does follow, which the CPython objects have
            // no equivalent of.
            Ok(w_bool_from(majit_gc::is_tracked(majit_ir::GcRef(args[0] as usize))))
        },
        "get_rpy_memory_usage" / 1 = |args| {
            // referents.py:97-104 / inspector.py:76-77.  The size is just the
            // translated object itself: no GC header and no reachable
            // internal storage.
            let _roots = pyre_object::gc_roots::push_roots();
            let obj_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(args[0]);
            let raw = gcref::unwrap(pyre_object::gc_roots::shadow_stack_get(obj_slot));
            let size = majit_gc::get_rpy_memory_usage(raw).ok_or_else(|| {
                    crate::PyError::not_implemented("operation not implemented by this GC")
                })?;
            Ok(w_int_new(size as i64))
        },
        "get_rpy_type_index" / 1 = |args| {
            // referents.py:106-115: a positive index into the translated
            // type-info group (index zero is the upstream dummy member).
            let _roots = pyre_object::gc_roots::push_roots();
            let obj_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(args[0]);
            let raw = gcref::unwrap(pyre_object::gc_roots::shadow_stack_get(obj_slot));
            let index = majit_gc::get_rpy_type_index(raw).ok_or_else(|| {
                    crate::PyError::not_implemented("operation not implemented by this GC")
                })?;
            Ok(w_int_new(index as i64))
        },
        "_dump_rpy_heap" / 1 = |args| {
            let fd = crate::baseobjspace::int_w(args[0])? as i32;
            dump_rpy_heap_fd(fd)?;
            Ok(w_none())
        },
        "get_typeids_z" / 0 = |_| {
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&typeids_z_bytes()?))
        },
        "get_typeids_list" / 0 = |_| {
            let list = majit_gc::get_typeids_list().ok_or_else(|| {
                crate::PyError::not_implemented("operation not implemented by this GC")
            })?;
            // Each `w_int_new` can collect, so pin as we go: an int built by an
            // earlier iteration would otherwise live only in a `Vec`.
            let _roots = pyre_object::gc_roots::push_roots();
            let first = pyre_object::gc_roots::shadow_stack_len();
            for value in list {
                pyre_object::gc_roots::pin_root(w_int_new(value as i64));
            }
            Ok(list_from_roots(first))
        },
        "is_finalized"  / 1 = |_| Ok(w_bool_from(false)),
        // CPython 3.14 `gc.freeze()` moves the surviving objects into a
        // permanent generation that later collections skip; it is a pre-fork
        // hint, not a semantic guarantee. The collector has no permanent
        // generation, so freezing and unfreezing are no-ops and the frozen
        // count is always the truthful zero.
        "freeze"           / 0 = |_| Ok(w_none()),
        "unfreeze"         / 0 = |_| Ok(w_none()),
        "get_freeze_count" / 0 = |_| Ok(w_int_new(0)),
    },
}
