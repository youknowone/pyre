//! gc module — PyPy: `pypy/module/gc/`.
//!
//! This file is the declarative table `moduledef.py` holds; the bodies live
//! in the submodules upstream splits them into: `interp_gc` (collection
//! control and the finalizer lock), `referents` (raw-heap inspection),
//! `app_referents` (the app-level `GcStats` and `dump_rpy_heap`) and `hook`
//! (`gc.hooks` and the stats objects it reports). Explicit collection runs
//! the complete RPython collection, then drains the finalizer queue
//! synchronously.
//!
//! Part of this module answers to 3.14 alone. `moduledef.py` binds no
//! `get_count`, `set_threshold`/`get_threshold`, `set_debug`/`get_debug` or
//! `freeze`/`unfreeze`/`get_freeze_count`, so those have no implementation to
//! follow and each one in `interp_gc` states what it answers and why. Nothing
//! grades them either: `test_gc` is an implementation-detail module on both
//! axes — `lib-python/conftest.py`'s testmap skips it and
//! `cpython_tests/run.py` carries that skip forward — so the assertions that
//! would pin these live in `extra_tests/snippets/` instead.

use pyre_object::*;

pub mod app_referents;
pub mod hook;
pub mod interp_gc;
pub mod referents;

use referents::{gcref, stats};

crate::py_module! {
    "gc",
    interpleveldefs: {
        // No `callbacks`.  `moduledef.py` defines none, and the collector-side
        // hook contract keeps its calls allocation-free, so nothing here can
        // run an app-level callback around a collection the way
        // `invoke_gc_callback` does.  Binding an empty list would satisfy
        // `hasattr` and then never call it, which is a silent failure where the
        // missing attribute is a loud one; `gc.hooks` is the notification
        // surface that does fire.
        "garbage"             => w_list_new(vec![]),
        "DEBUG_STATS"         => w_int_new(1),
        "DEBUG_COLLECTABLE"   => w_int_new(2),
        "DEBUG_UNCOLLECTABLE" => w_int_new(4),
        "DEBUG_SAVEALL"       => w_int_new(32),
        "DEBUG_LEAK"          => w_int_new(38),
        "GcCollectStepStats"  => hook::gc_collect_step_stats_type(),
        "GcRef"               => gcref::type_object(),
        "hooks"               => hook::hooks_object(),
    },
    inline_functions: {
        fn collect(
            #[default(w_int_new(interp_gc::NUM_GENERATIONS - 1))] generation: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            interp_gc::collect(generation)
        }

        fn collect_step() -> Result<PyObjectRef, crate::PyError> {
            interp_gc::collect_step()
        }

        fn enable_finalizers() -> Result<PyObjectRef, crate::PyError> {
            interp_gc::enable_finalizers()?;
            Ok(w_none())
        }

        fn disable_finalizers() -> Result<PyObjectRef, crate::PyError> {
            interp_gc::disable_finalizers();
            Ok(w_none())
        }

        fn get_objects(
            #[default(w_none())] generation: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            referents::get_objects(generation)
        }

        fn _get_stats(
            #[default(w_bool_from(false))] memory_pressure: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // referents.py `@unwrap_spec(memory_pressure=bool)`.
            Ok(stats::new(crate::baseobjspace::is_true(memory_pressure)?))
        }

        fn get_stats(
            #[default(w_bool_from(false))] memory_pressure: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            app_referents::new_public_gc_stats(crate::baseobjspace::is_true(memory_pressure)?)
        }

        fn dump_rpy_heap(file: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            app_referents::dump_rpy_heap_public(file)
        }
    },
    functions: {
        "disable"              / 0 = interp_gc::disable,
        "enable"               / 0 = interp_gc::enable,
        "isenabled"            / 0 = interp_gc::isenabled,
        "get_referrers"        / * = referents::get_referrers,
        "get_referents"        / * = referents::get_referents,
        "get_rpy_roots"        / 0 = referents::get_rpy_roots,
        "get_rpy_referents"    / 1 = referents::get_rpy_referents,
        "set_threshold"        / * = interp_gc::set_threshold,
        "get_threshold"        / 0 = interp_gc::get_threshold,
        "get_count"            / 0 = interp_gc::get_count,
        "get_debug"            / 0 = interp_gc::get_debug,
        "set_debug"            / 1 = interp_gc::set_debug,
        "is_tracked"           / 1 = interp_gc::is_tracked,
        "get_rpy_memory_usage" / 1 = referents::get_rpy_memory_usage,
        "get_rpy_type_index"   / 1 = referents::get_rpy_type_index,
        "_dump_rpy_heap"       / 1 = referents::_dump_rpy_heap,
        "get_typeids_z"        / 0 = referents::get_typeids_z,
        "get_typeids_list"     / 0 = referents::get_typeids_list,
        "is_finalized"         / 1 = interp_gc::is_finalized,
        "freeze"               / 0 = interp_gc::freeze,
        "unfreeze"             / 0 = interp_gc::unfreeze,
        "get_freeze_count"     / 0 = interp_gc::get_freeze_count,
    },
}
