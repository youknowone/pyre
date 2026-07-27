//! gc module — PyPy: `pypy/module/gc/`.
//!
//! Partial port of `interp_gc.py`. Explicit collection runs the complete
//! RPython collection, then drains the finalizer queue synchronously.

use pyre_object::*;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

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
    inline_functions: {
        fn collect(
            #[default(w_int_new(2))] generation: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // PyPy `interp_gc.py:7-26 collect` unwraps the optional generation
            // as an int before entering the body.  CPython 3.14 gives it the
            // default 2 and validates the three surviving generations.
            let generation = crate::baseobjspace::int_w(
                crate::baseobjspace::space_index(generation)?,
            )?;
            if !(0..=2).contains(&generation) {
                return Err(crate::PyError::value_error("invalid generation"));
            }
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
        }

        fn get_objects(
            #[default(w_none())] generation: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // CPython 3.14 `gc.get_objects(generation=None)`: -1/None means
            // every generation; 0 through 2 select a generation.
            let generation = if unsafe { is_none(generation) } {
                -1
            } else {
                crate::baseobjspace::int_w(crate::baseobjspace::space_index(generation)?)?
            };
            if generation < -1 {
                return Err(crate::PyError::value_error(
                    "generation parameter cannot be negative",
                ));
            }
            if generation > 2 {
                return Err(crate::PyError::value_error(
                    "generation parameter must be less than the number of available generations (3)",
                ));
            }
            let _roots = pyre_object::gc_roots::push_roots();
            let first = pyre_object::gc_roots::shadow_stack_len();
            majit_gc::get_objects(generation as i8, pin_object);
            let objects = (first..pyre_object::gc_roots::shadow_stack_len())
                .map(pyre_object::gc_roots::shadow_stack_get)
                .collect();
            Ok(w_list_new_object(objects))
        }
    },
    functions: {
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
            // An object referring to the same argument twice is reported once,
            // but one passed the same argument twice is reported twice.
            let _roots = pyre_object::gc_roots::push_roots();
            let all_first = pyre_object::gc_roots::shadow_stack_len();
            majit_gc::get_objects(-1, pin_object);
            let all_last = pyre_object::gc_roots::shadow_stack_len();
            let mut result = Vec::new();
            for slot in all_first..all_last {
                let w_obj = pyre_object::gc_roots::shadow_stack_get(slot);
                let _refs = pyre_object::gc_roots::push_roots();
                let refs_first = pyre_object::gc_roots::shadow_stack_len();
                pin_referents(w_obj);
                let refs_last = pyre_object::gc_roots::shadow_stack_len();
                for &w_arg in args {
                    if (refs_first..refs_last)
                        .any(|s| pyre_object::gc_roots::shadow_stack_get(s) == w_arg)
                    {
                        result.push(w_obj);
                    }
                }
            }
            // Every entry is also pinned in `all_first..all_last`, so the list
            // allocation below cannot leave one unrooted.
            Ok(w_list_new_object(result))
        },
        "get_referents" / * = |args| {
            // referents.py:128-145 get_referents.
            let _roots = pyre_object::gc_roots::push_roots();
            let first = pyre_object::gc_roots::shadow_stack_len();
            for &w_obj in args {
                pin_referents(w_obj);
            }
            let result = (first..pyre_object::gc_roots::shadow_stack_len())
                .map(pyre_object::gc_roots::shadow_stack_get)
                .collect();
            Ok(w_list_new_object(result))
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
            // value is validated but not kept.  The index protocol, so an
            // object carrying only `__int__` is a TypeError.
            let mut given = Vec::with_capacity(positional.len());
            for &w_value in positional {
                given.push(crate::baseobjspace::int_w(
                    crate::baseobjspace::space_index(w_value)?,
                )?);
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
