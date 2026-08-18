//! `_lsprof` module — PyPy: `pypy/module/_lsprof/interp_lsprof.py`.
//!
//! `moduledef.py` publishes only `Profiler`; the stats entry objects are
//! returned by `Profiler.getstats()` but are not bound in the module namespace.

use pyre_object::*;
use rustpython_wtf8::{Wtf8, Wtf8Buf};

use std::sync::OnceLock;
use std::time::Instant;

/// `interp_lsprof.py:305 W_Profiler`.  The entry trees and the context stack
/// belong to the wrapper object.  The mapdict prefix is required because
/// `cProfile.Profile` subclasses the type.
#[crate::pyre_class("_lsprof.Profiler")]
#[derive(Default)]
pub struct W_Profiler {
    pub map: *const u8,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    subcalls: bool,
    builtins: bool,
    current_context: Option<Box<ProfilerContext>>,
    w_callable: PyObjectRef,
    time_unit: f64,
    entries: Vec<ProfilerEntry>,
    builtin_entries: Vec<BuiltinEntry>,
    is_enabled: bool,
    total_timestamp: i64,
    total_real_time: f64,
}

const _: () = assert!(
    std::mem::offset_of!(W_Profiler, map)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, map),
    "W_Profiler must keep W_ObjectObject's map offset"
);
const _: () = assert!(
    std::mem::offset_of!(W_Profiler, storage)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, storage),
    "W_Profiler must keep W_ObjectObject's storage offset"
);

/// `interp_lsprof.py:44 W_StatsEntry`.  Its typedef publishes no `__new__`, so
/// no subclass of it can be instantiated and it carries no mapdict prefix; the
/// two references below are ordinary inline `gc_ptr_offsets` edges.
#[crate::pyre_class("_lsprof.profiler_entry")]
pub struct W_StatsEntry {
    frame: PyObjectRef,
    callcount: i64,
    reccallcount: i64,
    tt: f64,
    it: f64,
    w_calls: PyObjectRef,
}

/// `interp_lsprof.py:78 W_StatsSubEntry`, likewise not instantiable and
/// therefore prefix-free.
#[crate::pyre_class("_lsprof.profiler_subentry")]
pub struct W_StatsSubEntry {
    frame: PyObjectRef,
    callcount: i64,
    reccallcount: i64,
    tt: f64,
    it: f64,
}

#[derive(Clone)]
struct ProfilerSubEntry {
    frame: PyObjectRef,
    ll_tt: i64,
    ll_it: i64,
    callcount: i64,
    recursivecallcount: i64,
    recursion_level: i64,
}

struct ProfilerEntry {
    frame: PyObjectRef,
    ll_tt: i64,
    ll_it: i64,
    callcount: i64,
    recursivecallcount: i64,
    recursion_level: i64,
    calls: Vec<ProfilerSubEntry>,
}

struct BuiltinEntry {
    w_func: PyObjectRef,
    w_type: PyObjectRef,
    entry_index: usize,
}

struct ProfilerContext {
    entry_index: usize,
    ll_subt: i64,
    previous: Option<Box<ProfilerContext>>,
    ll_t0: i64,
}

fn profiler_clock() -> &'static Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now)
}

fn read_timestamp() -> i64 {
    // Composed from the seconds/subsecond pair rather than `Duration::as_nanos`,
    // whose `u128` result the codewriter cannot give a kind to.  The saturating
    // seconds term keeps the nanosecond product inside `i64` for the ~292 years
    // the type can name; past that the counter stops advancing rather than
    // wrapping into a negative interval.
    let elapsed = profiler_clock().elapsed();
    let seconds = elapsed.as_secs().min(i64::MAX as u64 / 1_000_000_000) as i64;
    seconds * 1_000_000_000 + i64::from(elapsed.subsec_nanos())
}

fn read_real_time() -> f64 {
    profiler_clock().elapsed().as_secs_f64()
}

fn timer_factor_for_external_timer(time_unit: f64) -> f64 {
    if time_unit > 0.0 {
        time_unit
    } else {
        1.0 / i64::MAX as f64
    }
}

impl ProfilerSubEntry {
    fn new(frame: PyObjectRef) -> Self {
        Self {
            frame,
            ll_tt: 0,
            ll_it: 0,
            callcount: 0,
            recursivecallcount: 0,
            recursion_level: 0,
        }
    }

    fn stats(&self, factor: f64) -> PyObjectRef {
        let _roots = pyre_object::gc_roots::push_roots();
        let frame_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(self.frame);
        W_StatsSubEntry::allocate_stable(W_StatsSubEntry {
            ob: PyObject::default(),
            frame: pyre_object::gc_roots::shadow_stack_get(frame_slot),
            callcount: self.callcount,
            reccallcount: self.recursivecallcount,
            tt: factor * self.ll_tt as f64,
            it: factor * self.ll_it as f64,
        })
    }

    fn stop(&mut self, tt: i64, it: i64) {
        if self.recursion_level > 0 {
            self.recursion_level -= 1;
        }
        if self.recursion_level == 0 {
            self.ll_tt += tt;
        } else {
            self.recursivecallcount += 1;
        }
        self.ll_it += it;
        self.callcount += 1;
    }
}

impl ProfilerEntry {
    fn new(frame: PyObjectRef) -> Self {
        Self {
            frame,
            ll_tt: 0,
            ll_it: 0,
            callcount: 0,
            recursivecallcount: 0,
            recursion_level: 0,
            calls: Vec::new(),
        }
    }

    fn stats(&self, factor: f64) -> PyObjectRef {
        let _roots = pyre_object::gc_roots::push_roots();
        let frame_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(self.frame);
        let w_sublist = if self.calls.is_empty() {
            w_none()
        } else {
            let root_base = pyre_object::gc_roots::shadow_stack_len();
            for subentry in &self.calls {
                pyre_object::gc_roots::pin_root(subentry.stats(factor));
            }
            let items = (root_base..pyre_object::gc_roots::shadow_stack_len())
                .map(pyre_object::gc_roots::shadow_stack_get)
                .collect();
            pyre_object::w_list_new(items)
        };
        // Take the sublist's own slot rather than assuming it follows the frame:
        // each subentry pinned above sits between them, so `frame_slot + 1` is
        // the first subentry whenever there is one.
        let sublist_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_sublist);
        W_StatsEntry::allocate_stable(W_StatsEntry {
            ob: PyObject::default(),
            frame: pyre_object::gc_roots::shadow_stack_get(frame_slot),
            callcount: self.callcount,
            reccallcount: self.recursivecallcount,
            tt: factor * self.ll_tt as f64,
            it: factor * self.ll_it as f64,
            w_calls: pyre_object::gc_roots::shadow_stack_get(sublist_slot),
        })
    }

    fn get_or_make_subentry(
        &mut self,
        frame: PyObjectRef,
        make: bool,
    ) -> Option<&mut ProfilerSubEntry> {
        if let Some(index) = self
            .calls
            .iter()
            .position(|subentry| subentry.frame == frame)
        {
            return self.calls.get_mut(index);
        }
        if make {
            self.calls.push(ProfilerSubEntry::new(frame));
            return self.calls.last_mut();
        }
        None
    }

    fn stop(&mut self, tt: i64, it: i64) {
        if self.recursion_level > 0 {
            self.recursion_level -= 1;
        }
        if self.recursion_level == 0 {
            self.ll_tt += tt;
        } else {
            self.recursivecallcount += 1;
        }
        self.ll_it += it;
        self.callcount += 1;
    }
}

fn returns_code(w_frame: PyObjectRef) -> PyObjectRef {
    w_frame
}

/// The `("<frame>", callcount, reccallcount, tt, it` opening both stats reprs
/// share, up to but not including the closing paren.
///
/// The two numeric pairs arrive already rendered.  A helper taking `tt`/`it` as
/// `f64` alongside the frame and both counts is a six-argument float-bearing
/// residual call, and the blackhole's dispatch table enumerates float-bearing
/// signatures only to five — a deopt through it would panic.
fn stats_repr_open(
    w_frame: PyObjectRef,
    callcount: i64,
    reccallcount: i64,
    tt: &str,
    it: &str,
) -> Result<Wtf8Buf, crate::PyError> {
    let frame_repr = unsafe { crate::display::py_repr_wtf8(w_frame)? };
    Ok(crate::display::wtf8_format!(
        "(\"",
        frame_repr,
        "\", ",
        callcount.to_string(),
        ", ",
        reccallcount.to_string(),
        ", ",
        tt,
        ", ",
        it
    ))
}

mod stats_entry_methods {
    use super::*;

    #[crate::pyre_methods]
    impl W_StatsEntry {
        #[getter]
        fn code(&self) -> PyObjectRef {
            returns_code(self.frame)
        }

        #[getter]
        fn callcount(&self) -> i64 {
            self.callcount
        }

        #[getter]
        fn reccallcount(&self) -> i64 {
            self.reccallcount
        }

        #[getter]
        fn inlinetime(&self) -> f64 {
            self.it
        }

        #[getter]
        fn totaltime(&self) -> f64 {
            self.tt
        }

        #[getter]
        fn calls(&self) -> PyObjectRef {
            self.w_calls
        }

        fn __repr__(&self) -> Result<PyObjectRef, crate::PyError> {
            let open = stats_repr_open(
                self.frame,
                self.callcount,
                self.reccallcount,
                &format!("{:.6}", self.tt),
                &format!("{:.6}", self.it),
            )?;
            let calls_repr = if unsafe { pyre_object::is_none(self.w_calls) } {
                Wtf8Buf::from_string("None".to_string())
            } else {
                unsafe { crate::display::py_repr_wtf8(self.w_calls)? }
            };
            Ok(pyre_object::w_str_from_wtf8_managed(
                crate::display::wtf8_format!(open, ", ", calls_repr, ")"),
            ))
        }
    }
}

mod stats_subentry_methods {
    use super::*;

    #[crate::pyre_methods]
    impl W_StatsSubEntry {
        #[getter]
        fn code(&self) -> PyObjectRef {
            returns_code(self.frame)
        }

        #[getter]
        fn callcount(&self) -> i64 {
            self.callcount
        }

        #[getter]
        fn reccallcount(&self) -> i64 {
            self.reccallcount
        }

        #[getter]
        fn inlinetime(&self) -> f64 {
            self.it
        }

        #[getter]
        fn totaltime(&self) -> f64 {
            self.tt
        }

        fn __repr__(&self) -> Result<PyObjectRef, crate::PyError> {
            let open = stats_repr_open(
                self.frame,
                self.callcount,
                self.reccallcount,
                &format!("{:.6}", self.tt),
                &format!("{:.6}", self.it),
            )?;
            Ok(pyre_object::w_str_from_wtf8_managed(
                crate::display::wtf8_format!(open, ")"),
            ))
        }
    }
}

fn module_name_wtf8(w_module: PyObjectRef) -> Option<Wtf8Buf> {
    if w_module.is_null()
        || unsafe { pyre_object::is_none(w_module) || !pyre_object::is_str(w_module) }
    {
        None
    } else {
        Some(unsafe { pyre_object::w_str_get_wtf8(w_module) }.to_wtf8_buf())
    }
}

fn create_spec_for_method(w_function: PyObjectRef, w_type: PyObjectRef) -> PyObjectRef {
    let name = if !w_function.is_null() && unsafe { crate::function::is_function(w_function) } {
        unsafe { crate::function::function_get_name(w_function) }
    } else {
        "?"
    };
    let class_name = if !w_function.is_null()
        && unsafe { crate::function::is_function(w_function) }
        && !w_type.is_null()
        && unsafe { pyre_object::is_type(w_type) }
    {
        unsafe { crate::baseobjspace::lookup_where_pair(w_type, name) }
            .map(|(w_realclass, _)| unsafe { pyre_object::w_type_get_name(w_realclass) })
            .unwrap_or_else(|| unsafe { pyre_object::w_type_get_name(w_type) })
    } else if !w_type.is_null() && unsafe { pyre_object::is_type(w_type) } {
        unsafe { pyre_object::w_type_get_name(w_type) }
    } else {
        "object"
    };
    pyre_object::w_str_new(&format!("<method '{name}' of '{class_name}' objects>"))
}

fn create_spec_for_function(w_func: PyObjectRef) -> PyObjectRef {
    let name = unsafe { crate::function::function_get_name(w_func) };
    let w_module = unsafe { (*(w_func as *const crate::function::Function)).w_module };
    let text = if unsafe { crate::function::function_has_builtin_code(w_func) } {
        if let Some(module) = module_name_wtf8(w_module) {
            crate::display::wtf8_format!("<built-in method ", module, ".", name, ">")
        } else {
            crate::display::wtf8_format!("<built-in function ", name, ">")
        }
    } else if let Some(module) = module_name_wtf8(w_module) {
        crate::display::wtf8_format!("<", module, ".", name, ">")
    } else {
        crate::display::wtf8_format!("<", name, ">")
    };
    pyre_object::w_str_from_wtf8_managed(text)
}

fn create_spec_for_object(w_type: PyObjectRef) -> PyObjectRef {
    let class_name = if !w_type.is_null() && unsafe { pyre_object::is_type(w_type) } {
        unsafe { pyre_object::w_type_get_name(w_type) }
    } else {
        "object"
    };
    pyre_object::w_str_new(&format!("<'{class_name}' object>"))
}

fn prepare_spec(w_arg: PyObjectRef) -> (PyObjectRef, PyObjectRef, PyObjectRef) {
    if !w_arg.is_null() && unsafe { pyre_object::function::is_method(w_arg) } {
        let w_func = unsafe { pyre_object::function::w_method_get_func(w_arg) };
        let w_self = unsafe { pyre_object::function::w_method_get_self(w_arg) };
        let w_type = crate::typedef::r#type(w_self).map_or(pyre_object::PY_NULL, |ty| ty.as_ptr());
        let w_frame = create_spec_for_method(w_func, w_type);
        (w_func, w_type, w_frame)
    } else if !w_arg.is_null() && unsafe { crate::function::is_function(w_arg) } {
        let w_frame = create_spec_for_function(w_arg);
        (w_arg, pyre_object::PY_NULL, w_frame)
    } else {
        let w_type = crate::typedef::r#type(w_arg).map_or(pyre_object::PY_NULL, |ty| ty.as_ptr());
        let w_frame = create_spec_for_object(w_type);
        (pyre_object::PY_NULL, w_type, w_frame)
    }
}

fn lsprof_call(
    _space: PyObjectRef,
    w_self: PyObjectRef,
    frame: *mut crate::pyframe::PyFrame,
    event: &str,
    w_arg: PyObjectRef,
) -> Result<(), crate::PyError> {
    let Some(profiler) = W_Profiler::from_obj(w_self) else {
        return Ok(());
    };
    match event {
        "call" => {
            if !frame.is_null() {
                profiler.enter_call(unsafe { (*frame).fget_f_code() });
            }
        }
        "return" => {
            if !frame.is_null() {
                profiler.enter_return(unsafe { (*frame).fget_f_code() });
            }
        }
        "c_call" => {
            if profiler.builtins {
                profiler.enter_builtin_call(w_self, w_arg);
            }
        }
        "c_return" | "c_exception" => {
            if profiler.builtins {
                profiler.enter_builtin_return(w_arg);
            }
        }
        _ => {}
    }
    Ok(())
}

impl W_Profiler {
    fn ll_timer(&self) -> i64 {
        if self.w_callable.is_null() {
            return read_timestamp();
        }
        match crate::call::call_function_impl_result(self.w_callable, &[])
            .and_then(crate::baseobjspace::int_w)
        {
            Ok(value) => value,
            Err(mut err) => {
                err.write_unraisable(w_none(), Wtf8::new("timer function "), self.w_callable);
                0
            }
        }
    }

    fn write_barrier(&self) {
        let self_obj = self as *const Self as PyObjectRef;
        pyre_object::gc_hook::try_gc_write_barrier(self_obj as *mut u8);
    }

    fn get_or_make_entry(&mut self, frame: PyObjectRef, make: bool) -> Option<usize> {
        if let Some(index) = self.entries.iter().position(|entry| entry.frame == frame) {
            return Some(index);
        }
        if make {
            self.write_barrier();
            self.entries.push(ProfilerEntry::new(frame));
            return Some(self.entries.len() - 1);
        }
        None
    }

    fn get_or_make_builtin_entry(
        &mut self,
        w_func: PyObjectRef,
        w_type: PyObjectRef,
        w_frame: PyObjectRef,
        make: bool,
    ) -> Option<usize> {
        if let Some(entry) = self
            .builtin_entries
            .iter()
            .find(|entry| entry.w_func == w_func && entry.w_type == w_type)
        {
            return Some(entry.entry_index);
        }
        if make {
            self.write_barrier();
            self.entries.push(ProfilerEntry::new(w_frame));
            let entry_index = self.entries.len() - 1;
            self.builtin_entries.push(BuiltinEntry {
                w_func,
                w_type,
                entry_index,
            });
            return Some(entry_index);
        }
        None
    }

    fn enter_context(&mut self, entry_index: usize) {
        let previous = self.current_context.take();
        self.entries[entry_index].recursion_level += 1;
        if self.subcalls
            && let Some(previous_context) = previous.as_ref()
        {
            let caller_index = previous_context.entry_index;
            let frame = self.entries[entry_index].frame;
            self.write_barrier();
            if let Some(subentry) = self.entries[caller_index].get_or_make_subentry(frame, true) {
                subentry.recursion_level += 1;
            }
        }
        let ll_t0 = self.ll_timer();
        self.current_context = Some(Box::new(ProfilerContext {
            entry_index,
            ll_subt: 0,
            previous,
            ll_t0,
        }));
    }

    fn stop_context(&mut self, context: &mut ProfilerContext, entry_index: usize) {
        let tt = self.ll_timer() - context.ll_t0;
        let it = tt - context.ll_subt;
        if let Some(previous) = context.previous.as_mut() {
            previous.ll_subt += tt;
        }
        self.entries[entry_index].stop(tt, it);
        if self.subcalls
            && let Some(previous) = context.previous.as_ref()
        {
            let caller_index = previous.entry_index;
            let frame = self.entries[entry_index].frame;
            if let Some(subentry) = self.entries[caller_index].get_or_make_subentry(frame, false) {
                subentry.stop(tt, it);
            }
        }
    }

    fn enter_call(&mut self, f_code: PyObjectRef) {
        if let Some(entry_index) = self.get_or_make_entry(f_code, true) {
            self.enter_context(entry_index);
        }
    }

    fn enter_return(&mut self, f_code: PyObjectRef) {
        let Some(mut context) = self.current_context.take() else {
            return;
        };
        if let Some(entry_index) = self.get_or_make_entry(f_code, false) {
            self.stop_context(&mut context, entry_index);
        }
        self.current_context = context.previous.take();
    }

    fn enter_builtin_call(&mut self, self_obj: PyObjectRef, w_arg: PyObjectRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(self_obj);
        pyre_object::gc_roots::pin_root(w_arg);
        let (w_func, w_type, w_frame) =
            prepare_spec(pyre_object::gc_roots::shadow_stack_get(root_base + 1));
        pyre_object::gc_roots::pin_root(w_frame);
        let this = W_Profiler::from_obj(pyre_object::gc_roots::shadow_stack_get(root_base))
            .expect("profile hook owns a live Profiler");
        if let Some(entry_index) = this.get_or_make_builtin_entry(
            w_func,
            w_type,
            pyre_object::gc_roots::shadow_stack_get(root_base + 2),
            true,
        ) {
            this.enter_context(entry_index);
        }
    }

    fn enter_builtin_return(&mut self, w_arg: PyObjectRef) {
        let Some(mut context) = self.current_context.take() else {
            return;
        };
        let (w_func, w_type, w_frame) = prepare_spec(w_arg);
        if let Some(entry_index) = self.get_or_make_builtin_entry(w_func, w_type, w_frame, false) {
            self.stop_context(&mut context, entry_index);
        }
        self.current_context = context.previous.take();
    }

    fn flush_unmatched(&mut self) {
        let mut context = self.current_context.take();
        while let Some(mut active) = context {
            let entry_index = active.entry_index;
            self.stop_context(&mut active, entry_index);
            context = active.previous.take();
        }
    }

    fn stats(&self, factor: f64) -> PyObjectRef {
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        for entry in &self.entries {
            if entry.callcount != 0 {
                pyre_object::gc_roots::pin_root(entry.stats(factor));
            }
        }
        let items = (root_base..pyre_object::gc_roots::shadow_stack_len())
            .map(pyre_object::gc_roots::shadow_stack_get)
            .collect();
        pyre_object::w_list_new(items)
    }

    fn walk_owned_refs(&mut self, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
        fn visit(slot: &mut PyObjectRef, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
            if !slot.is_null() {
                f(slot as *mut PyObjectRef as *mut majit_ir::GcRef);
            }
        }

        visit(&mut self.w_callable, f);
        for entry in &mut self.entries {
            visit(&mut entry.frame, f);
            for subentry in &mut entry.calls {
                visit(&mut subentry.frame, f);
            }
        }
        for entry in &mut self.builtin_entries {
            visit(&mut entry.w_func, f);
            visit(&mut entry.w_type, f);
        }
    }
}

mod profiler_methods {
    use super::*;

    #[crate::pyre_methods]
    impl W_Profiler {
        #[staticmethod]
        fn __new__(
            cls: PyObjectRef,
            #[default(w_none())] w_callable: PyObjectRef,
            #[default(0.0f64)] time_unit: f64,
            #[default(1i32)] subcalls: i32,
            #[default(1i32)] builtins: i32,
        ) -> Result<PyObjectRef, crate::PyError> {
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let callable = if unsafe { pyre_object::is_none(w_callable) } {
                pyre_object::PY_NULL
            } else {
                w_callable
            };
            let _roots = pyre_object::gc_roots::push_roots();
            let callable_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(callable);
            let obj = W_Profiler::allocate_stable(W_Profiler {
                ob: PyObject::default(),
                subcalls: subcalls != 0,
                builtins: builtins != 0,
                w_callable: pyre_object::gc_roots::shadow_stack_get(callable_slot),
                time_unit,
                ..W_Profiler::default()
            });
            unsafe { (*obj).w_class = cls };
            Ok(obj)
        }

        fn enable(
            &mut self,
            #[default(w_none())] w_subcalls: PyObjectRef,
            #[default(w_none())] w_builtins: PyObjectRef,
        ) -> Result<(), crate::PyError> {
            if self.is_enabled {
                return Ok(());
            }
            // `_lsprof.c profiler_enable` claims the profiler tool id before
            // touching any of its own state, so a second profiler's `enable`
            // reports the conflict and leaves the first one installed.
            crate::module::sys::vm::monitoring_use_tool_id(
                crate::module::sys::vm::MONITORING_PROFILER_ID,
                w_str_new("cProfile"),
            )?;
            if !unsafe { pyre_object::is_none(w_subcalls) } {
                self.subcalls = crate::baseobjspace::is_true(w_subcalls)?;
            }
            if !unsafe { pyre_object::is_none(w_builtins) } {
                self.builtins = crate::baseobjspace::is_true(w_builtins)?;
            }
            self.is_enabled = true;
            self.total_real_time -= read_real_time();
            self.total_timestamp -= read_timestamp();
            let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
            unsafe {
                (*ec).setllprofile(Some(lsprof_call), self as *mut Self as PyObjectRef)?;
            }
            Ok(())
        }

        fn disable(&mut self) -> Result<(), crate::PyError> {
            if !self.is_enabled {
                return Ok(());
            }
            crate::module::sys::vm::monitoring_free_tool_id(
                crate::module::sys::vm::MONITORING_PROFILER_ID,
            );
            self.is_enabled = false;
            self.total_timestamp += read_timestamp();
            self.total_real_time += read_real_time();
            let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
            unsafe {
                (*ec).setllprofile(None, pyre_object::PY_NULL)?;
            }
            self.flush_unmatched();
            Ok(())
        }

        /// `_lsprof.c profiler_clear` — settle the contexts still on the stack,
        /// then drop the accumulated entries.  `interp_lsprof.py`'s typedef
        /// publishes no `clear`, but `cProfile` and `test_cprofile` both call
        /// it, so the method belongs on the type.
        fn clear(&mut self) {
            self.flush_unmatched();
            self.entries.clear();
            self.builtin_entries.clear();
        }

        fn getstats(&self) -> Result<PyObjectRef, crate::PyError> {
            let factor = if self.w_callable.is_null() {
                if self.is_enabled {
                    return Err(crate::PyError::runtime_error(
                        "Profiler instance must be disabled before getting the stats",
                    ));
                }
                if self.total_timestamp != 0 {
                    self.total_real_time / self.total_timestamp as f64
                } else {
                    1.0
                }
            } else {
                timer_factor_for_external_timer(self.time_unit)
            };
            Ok(self.stats(factor))
        }
    }
}

/// Drop the Rust-owned profiler bookkeeping.
///
/// # Safety
/// `obj` must be a GC-dead `W_Profiler`.
pub unsafe fn w_profiler_dealloc(obj: PyObjectRef) {
    unsafe {
        std::ptr::drop_in_place(obj as *mut W_Profiler);
    }
}

/// Walk `W_Profiler`'s indirect entry trees.
///
/// # Safety
/// `obj_addr` must point at a live `W_Profiler`.
pub unsafe fn w_profiler_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let profiler = unsafe { &mut *(obj_addr as *mut W_Profiler) };
    profiler.walk_owned_refs(f);
}

crate::py_module! {
    "_lsprof",
    interpleveldefs: {
        "Profiler" => profiler_methods::type_object(),
    },
    extra_init: |ns| {
        let _ = ns;
        let _ = stats_entry_methods::type_object();
        let _ = stats_subentry_methods::type_object();
    },
}
