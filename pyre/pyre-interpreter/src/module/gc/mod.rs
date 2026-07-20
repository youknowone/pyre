//! gc module — PyPy: `pypy/module/gc/`.
//!
//! Partial port of `interp_gc.py`. Explicit collection runs the complete
//! RPython collection, then drains the finalizer queue synchronously.

use pyre_object::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// `interp_gc.py` tracks a process-wide `enabled` flag on the GC
/// frontend; pyre has no generational threshold knob, but
/// `gc.isenabled()` should reflect the most recent `enable`/`disable`
/// call so callers that toggle and re-read the state stay consistent.
static GC_ENABLED: AtomicBool = AtomicBool::new(true);

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
            for obj in pending {
                action._call_finalizer(obj);
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
    },
    functions: {
        // `interp_gc.py:7-26 collect` — argument `generation` ignored per
        // upstream.
        "collect"       / 1 = |_| {
            crate::baseobjspace::clear_method_cache();
            crate::objspace::std::mapdict::clear_map_attr_cache();
            pyre_object::gc_hook::try_gc_collect();
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
            Ok(w_int_new(0))
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
        "get_objects"   / 1 = |_| Ok(w_list_new(vec![])),
        "get_referrers" / * = |_| Ok(w_list_new(vec![])),
        "get_referents" / * = |_| Ok(w_list_new(vec![])),
        "set_threshold" / 0 = |_| Ok(w_none()),
        "get_threshold" / 0 = |_| Ok(w_tuple_new(vec![
            w_int_new(700), w_int_new(10), w_int_new(10),
        ])),
        "get_count"     / 0 = |_| Ok(w_tuple_new(vec![
            w_int_new(0), w_int_new(0), w_int_new(0),
        ])),
        "is_tracked"    / 1 = |_| Ok(w_bool_from(false)),
        "is_finalized"  / 1 = |_| Ok(w_bool_from(false)),
        "freeze"        / 0 = |_| Ok(w_none()),
    },
}
