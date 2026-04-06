//! rpython/rlib/jit.py — User-facing JIT hint API.
//!
//! All functions in this module are no-ops at interpreter runtime.
//! When the `#[jit_interp]` proc macro processes user code, calls to these
//! functions are recognized and lowered to JitCode operations:
//!
//! - `promote(x)` → `int_guard_value(x)` / `ref_guard_value(x)` / `float_guard_value(x)`
//! - `assert_green(x)` → compile-time assertion
//! - `record_exact_class(v, cls)` → RECORD_EXACT_CLASS IR op
//! - `record_exact_value(v, c)` → RECORD_EXACT_VALUE IR op
//!
//! RPython reference: rpython/rlib/jit.py

// ── promote ──
// rlib/jit.py:100-124

/// Promotes a variable in a trace to a constant.
///
/// When a variable is promoted, a guard is inserted that assumes the value
/// of the variable is constant. In other words, the value of the variable
/// is checked to be the same as it was at trace collection time.  Once the
/// variable is assumed constant, more aggressive constant folding may be
/// possible.
///
/// If however, the guard fails frequently, a bridge will be generated
/// this time assuming the constancy of the variable under its new value.
/// This optimisation should be used carefully, as in extreme cases, where
/// the promoted variable is not very constant at all, code explosion can
/// occur. In turn this leads to poor performance.
///
/// Overpromotion is characterised by a cascade of bridges branching from
/// very similar guard_value opcodes, each guarding the same variable under
/// a different value.
///
/// rlib/jit.py:100 — `promote(x) = hint(x, promote=True)`
#[inline(always)]
pub fn promote<T: Copy>(x: T) -> T {
    x
}

// ── hint ──
// rlib/jit.py:79-97
//
// In RPython, hint(x, **kwds) accepts keyword arguments:
//   promote, promote_string, promote_unicode,
//   access_directly, fresh_virtualizable, force_virtualizable
//
// In Rust, keyword arguments don't exist.  The proc macro recognizes
// `promote(x)` directly.  Virtualizable hints are handled by the
// `virtualizable!` macro and `#[jit_interp]` storage configuration.
// This function exists for structural parity — it is a no-op.

/// Hint for the JIT.
///
/// rlib/jit.py:79 — `hint(x, **kwds)` returns x unchanged.
#[inline(always)]
pub fn hint<T>(x: T) -> T {
    x
}

// ── dont_look_inside, elidable, unroll_safe, loop_invariant ──
// These are decorators in RPython, implemented as proc macro attributes
// in majit-macros: #[elidable], #[dont_look_inside], #[unroll_safe],
// #[loop_invariant], #[not_in_trace].

// ── we_are_jitted ──
// rlib/jit.py:355-358

/// Considered as true during tracing and blackholing,
/// so its consequences are reflected into jitted code.
///
/// rlib/jit.py:355
#[inline(always)]
pub fn we_are_jitted() -> bool {
    majit_backend::we_are_jitted()
}

// ── isconstant ──
// rlib/jit.py:269-279

/// While tracing, returns whether or not the value is currently known to be
/// constant. This is not perfect, values can become constant later. Mostly for
/// use with look_inside_iff.
///
/// This is for advanced usage only.
///
/// rlib/jit.py:269
#[inline(always)]
pub fn isconstant<T>(_value: &T) -> bool {
    false
}

// ── isvirtual ──
// rlib/jit.py:281-292

/// Returns if this value is virtual, while tracing. Can be wrong in both
/// directions. It tries to be conservative by default, but can also sometimes
/// return True for something that does not end up completely virtual.
///
/// This is for advanced usage only.
///
/// rlib/jit.py:281
#[inline(always)]
pub fn isvirtual<T: ?Sized>(_value: &T) -> bool {
    false
}

// ── loop_unrolling_heuristic ──
// rlib/jit.py:294-301

/// In which cases iterating over items of lst can be unrolled.
///
/// rlib/jit.py:294
#[inline(always)]
pub fn loop_unrolling_heuristic<T>(_lst: &[T], size: usize, cutoff: usize) -> bool {
    size == 0 || (isconstant(&size) && (isvirtual(_lst) || size <= cutoff))
}

// ── current_trace_length ──
// rlib/jit.py:408-414

/// During JIT tracing, returns the current trace length (as a constant).
/// If not tracing, returns -1.
///
/// rlib/jit.py:408
#[inline(always)]
pub fn current_trace_length() -> i64 {
    -1
}

// ── jit_debug ──
// rlib/jit.py:416-421

/// When JITted, cause an extra operation JIT_DEBUG to appear in
/// the graphs.  Should not be left after debugging.
///
/// rlib/jit.py:416
#[inline(always)]
pub fn jit_debug(_string: &str, _arg1: i64, _arg2: i64, _arg3: i64, _arg4: i64) {}

// ── assert_green ──
// rlib/jit.py:423-428

/// Very strong assert: checks that 'value' is a green
/// (a JIT compile-time constant).
///
/// rlib/jit.py:423
#[inline(always)]
pub fn assert_green<T>(_value: &T) {}

// ── VRefs ──
// rlib/jit.py:460-524

/// Creates a 'vref' object that contains a reference to 'x'.  Calls
/// to virtual_ref/virtual_ref_finish must be properly nested.  The idea
/// is that the object 'x' is supposed to be JITted as a virtual between
/// the calls to virtual_ref and virtual_ref_finish, but the 'vref'
/// object can escape at any point in time.  If at runtime it is
/// dereferenced, it returns 'x', which is then forced.
///
/// rlib/jit.py:463
#[inline(always)]
pub fn virtual_ref<T>(x: &T) -> DirectJitVRef<T> {
    DirectJitVRef {
        _x: x as *const T,
        _state: VRefState::NonForced,
    }
}

/// See docstring in virtual_ref(x).
///
/// rlib/jit.py:475
#[inline(always)]
pub fn virtual_ref_finish<T>(vref: &mut DirectJitVRef<T>, _x: &T) {
    vref._finish();
}

/// Creates a 'vref' that just returns x when called; nothing more special.
/// Used for None or for frames outside JIT scope.
///
/// rlib/jit.py:482
#[inline(always)]
pub fn non_virtual_ref<T>(x: &T) -> DirectVRef<T> {
    DirectVRef {
        _x: x as *const T,
        _state: VRefState::NonForced,
    }
}

/// rlib/jit.py:495 — DirectVRef
#[derive(Debug)]
pub struct DirectVRef<T> {
    _x: *const T,
    _state: VRefState,
}

/// rlib/jit.py:517 — DirectJitVRef(DirectVRef)
#[derive(Debug)]
pub struct DirectJitVRef<T> {
    _x: *const T,
    _state: VRefState,
}

/// rlib/jit.py:498 — _state field
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VRefState {
    NonForced,
    Forced,
    Invalid,
}

/// rlib/jit.py:487
pub struct InvalidVirtualRef;

impl<T> DirectVRef<T> {
    /// rlib/jit.py:500 — __call__
    pub fn force(&mut self) -> Result<&T, InvalidVirtualRef> {
        match self._state {
            VRefState::NonForced => {
                self._state = VRefState::Forced;
                Ok(unsafe { &*self._x })
            }
            VRefState::Forced => Ok(unsafe { &*self._x }),
            VRefState::Invalid => Err(InvalidVirtualRef),
        }
    }

    /// rlib/jit.py:507 — virtual property
    pub fn virtual_(&self) -> bool {
        self._state == VRefState::NonForced
    }

    /// rlib/jit.py:513 — _finish
    fn _finish(&mut self) {
        if self._state == VRefState::NonForced {
            self._state = VRefState::Invalid;
        }
    }
}

impl<T> DirectJitVRef<T> {
    /// rlib/jit.py:500 — __call__
    pub fn force(&mut self) -> Result<&T, InvalidVirtualRef> {
        match self._state {
            VRefState::NonForced => {
                self._state = VRefState::Forced;
                Ok(unsafe { &*self._x })
            }
            VRefState::Forced => Ok(unsafe { &*self._x }),
            VRefState::Invalid => Err(InvalidVirtualRef),
        }
    }

    /// rlib/jit.py:507 — virtual property
    pub fn virtual_(&self) -> bool {
        self._state == VRefState::NonForced
    }

    /// rlib/jit.py:513 — _finish
    fn _finish(&mut self) {
        if self._state == VRefState::NonForced {
            self._state = VRefState::Invalid;
        }
    }
}

// ── JitHintError ──
// rlib/jit.py:559

/// Inconsistency in the JIT hints.
///
/// rlib/jit.py:559
#[derive(Debug)]
pub struct JitHintError(pub String);

impl std::fmt::Display for JitHintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JitHintError: {}", self.0)
    }
}

impl std::error::Error for JitHintError {}

// ── ENABLE_ALL_OPTS ──
// rlib/jit.py:562

pub const ENABLE_ALL_OPTS: &str = "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll";

// ── PARAMETERS ──
// rlib/jit.py:588-605

/// Default JIT parameters.
///
/// rlib/jit.py:588
pub const PARAMETERS: JitParameters = JitParameters {
    threshold: 1039,          // just above 1024, prime
    function_threshold: 1619, // slightly more than one above, also prime
    trace_eagerness: 200,
    decay: 40,
    trace_limit: 6000,
    inlining: true,
    loop_longevity: 1000,
    retrace_limit: 0,
    pureop_historylength: 16,
    max_retrace_guards: 15,
    max_unroll_loops: 0,
    disable_unrolling: 200,
    enable_opts: EnableOpts::All,
    max_unroll_recursion: 7,
    vec: false,
    vec_all: false,
    vec_cost: 0,
};

/// rlib/jit.py:588 — PARAMETERS dict
#[derive(Debug, Clone)]
pub struct JitParameters {
    pub threshold: u32,
    pub function_threshold: u32,
    pub trace_eagerness: u32,
    pub decay: u32,
    pub trace_limit: u32,
    pub inlining: bool,
    pub loop_longevity: u32,
    pub retrace_limit: u32,
    pub pureop_historylength: u32,
    pub max_retrace_guards: u32,
    pub max_unroll_loops: u32,
    pub disable_unrolling: u32,
    pub enable_opts: EnableOpts,
    pub max_unroll_recursion: u32,
    pub vec: bool,
    pub vec_all: bool,
    pub vec_cost: u32,
}

/// rlib/jit.py:562,600
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnableOpts {
    All,
    Custom(String),
}

// ── set_param / set_user_param ──
// rlib/jit.py:812-880

/// Set a JIT parameter by name.
///
/// rlib/jit.py:819
pub fn set_param(params: &mut JitParameters, name: &str, value: i64) {
    match name {
        "threshold" => params.threshold = value as u32,
        "function_threshold" => params.function_threshold = value as u32,
        "trace_eagerness" => params.trace_eagerness = value as u32,
        "decay" => params.decay = value as u32,
        "trace_limit" => params.trace_limit = value as u32,
        "inlining" => params.inlining = value != 0,
        "loop_longevity" => params.loop_longevity = value as u32,
        "retrace_limit" => params.retrace_limit = value as u32,
        "pureop_historylength" => params.pureop_historylength = value as u32,
        "max_retrace_guards" => params.max_retrace_guards = value as u32,
        "max_unroll_loops" => params.max_unroll_loops = value as u32,
        "disable_unrolling" => params.disable_unrolling = value as u32,
        "max_unroll_recursion" => params.max_unroll_recursion = value as u32,
        "vec" => params.vec = value != 0,
        "vec_all" => params.vec_all = value != 0,
        "vec_cost" => params.vec_cost = value as u32,
        _ => {}
    }
}

/// Set a JIT parameter to its default value.
///
/// rlib/jit.py:825
pub fn set_param_to_default(params: &mut JitParameters, name: &str) {
    let defaults = PARAMETERS;
    match name {
        "threshold" => params.threshold = defaults.threshold,
        "function_threshold" => params.function_threshold = defaults.function_threshold,
        "trace_eagerness" => params.trace_eagerness = defaults.trace_eagerness,
        "decay" => params.decay = defaults.decay,
        "trace_limit" => params.trace_limit = defaults.trace_limit,
        "inlining" => params.inlining = defaults.inlining,
        "loop_longevity" => params.loop_longevity = defaults.loop_longevity,
        "retrace_limit" => params.retrace_limit = defaults.retrace_limit,
        "pureop_historylength" => params.pureop_historylength = defaults.pureop_historylength,
        "max_retrace_guards" => params.max_retrace_guards = defaults.max_retrace_guards,
        "max_unroll_loops" => params.max_unroll_loops = defaults.max_unroll_loops,
        "disable_unrolling" => params.disable_unrolling = defaults.disable_unrolling,
        "max_unroll_recursion" => params.max_unroll_recursion = defaults.max_unroll_recursion,
        "vec" => params.vec = defaults.vec,
        "vec_all" => params.vec_all = defaults.vec_all,
        "vec_cost" => params.vec_cost = defaults.vec_cost,
        _ => {}
    }
}

/// Set parameters from a user string: 'param=value,param=value'
/// or 'off' to disable JIT.
///
/// rlib/jit.py:836
pub fn set_user_param(params: &mut JitParameters, text: &str) -> Result<(), JitHintError> {
    if text == "off" {
        params.threshold = u32::MAX;
        params.function_threshold = u32::MAX;
        return Ok(());
    }
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part == "default" {
            *params = PARAMETERS;
            continue;
        }
        let (name, val_str) = part
            .split_once('=')
            .ok_or_else(|| JitHintError(format!("malformed token: {part}")))?;
        let name = name.trim();
        if name == "enable_opts" {
            let val_str = val_str.trim();
            params.enable_opts = if val_str == "all" {
                EnableOpts::All
            } else {
                EnableOpts::Custom(val_str.to_string())
            };
            continue;
        }
        let value: i64 = val_str
            .trim()
            .parse()
            .map_err(|_| JitHintError(format!("invalid value for {name}: {}", val_str.trim())))?;
        set_param(params, name, value);
    }
    Ok(())
}

// ── record_exact_class ──
// rlib/jit.py:1181-1189

/// Assure the JIT that value is an instance of cls. This is a precise
/// class check, like a guard_class.
///
/// rlib/jit.py:1181
#[inline(always)]
pub fn record_exact_class<T>(_value: &T, _cls: usize) {}

// ── record_exact_value ──
// rlib/jit.py:1260-1265

/// Assure the JIT that value is the same as const_value.
///
/// rlib/jit.py:1260
#[inline(always)]
pub fn record_exact_value<T: Copy>(value: T, _const_value: T) -> T {
    value
}

// ── record_known_result ──
// rlib/jit.py:1224-1239

/// Assure the JIT that func(args) will produce result.
/// func must be an elidable function.
///
/// rlib/jit.py:1224
#[inline(always)]
pub fn record_known_result<T, F>(_result: T, _func: F) {}

// ── conditional_call ──
// rlib/jit.py:1300-1316

/// Does the same as:
///
///     if condition { function() }
///
/// but is better for the JIT, in case the condition is often false
/// but could be true occasionally.  It allows the JIT to always produce
/// bridge-free code.  The function is never looked inside.
///
/// rlib/jit.py:1300
#[inline(always)]
pub fn conditional_call<F: FnOnce()>(condition: bool, function: F) {
    if condition {
        function();
    }
}

// ── conditional_call_elidable ──
// rlib/jit.py:1321-1359

/// Does the same as:
///
///     match value {
///         Some(v) => v,
///         None => function(),
///     }
///
/// For the JIT.  Allows one branch which doesn't create a bridge,
/// typically used for caching.
///
/// rlib/jit.py:1321
#[inline(always)]
pub fn conditional_call_elidable<T, F: FnOnce() -> T>(value: Option<T>, function: F) -> T {
    match value {
        Some(v) => v,
        None => function(),
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_int() {
        assert_eq!(promote(42), 42);
    }

    #[test]
    fn test_promote_ref_copy() {
        let x: usize = 0xdead;
        assert_eq!(promote(x), 0xdead);
    }

    #[test]
    fn test_we_are_jitted_default() {
        assert!(!we_are_jitted());
    }

    #[test]
    fn test_isconstant_default() {
        assert!(!isconstant(&42));
    }

    #[test]
    fn test_isvirtual_default() {
        let v = vec![1, 2, 3];
        assert!(!isvirtual(&v));
    }

    #[test]
    fn test_current_trace_length_default() {
        assert_eq!(current_trace_length(), -1);
    }

    #[test]
    fn test_virtual_ref_lifecycle() {
        let val = 42i64;
        let mut vref = virtual_ref(&val);
        assert!(vref.virtual_());
        assert_eq!(*vref.force().unwrap(), 42);
        assert!(!vref.virtual_());
    }

    #[test]
    fn test_virtual_ref_finish_invalidates() {
        let val = 42i64;
        let mut vref = virtual_ref(&val);
        virtual_ref_finish(&mut vref, &val);
        assert!(vref.force().is_err());
    }

    #[test]
    fn test_parameters_default() {
        assert_eq!(PARAMETERS.threshold, 1039);
        assert_eq!(PARAMETERS.function_threshold, 1619);
        assert_eq!(PARAMETERS.trace_eagerness, 200);
        assert_eq!(PARAMETERS.trace_limit, 6000);
    }

    #[test]
    fn test_set_user_param() {
        let mut p = PARAMETERS;
        set_user_param(&mut p, "threshold=100,trace_limit=3000").unwrap();
        assert_eq!(p.threshold, 100);
        assert_eq!(p.trace_limit, 3000);
        // other params unchanged
        assert_eq!(p.function_threshold, 1619);
    }

    #[test]
    fn test_set_user_param_off() {
        let mut p = PARAMETERS;
        set_user_param(&mut p, "off").unwrap();
        assert_eq!(p.threshold, u32::MAX);
        assert_eq!(p.function_threshold, u32::MAX);
    }

    #[test]
    fn test_set_user_param_default() {
        let mut p = PARAMETERS;
        p.threshold = 999;
        set_user_param(&mut p, "default").unwrap();
        assert_eq!(p.threshold, 1039);
    }

    #[test]
    fn test_conditional_call() {
        let mut called = false;
        conditional_call(false, || {
            called = true;
        });
        assert!(!called);
        conditional_call(true, || {
            called = true;
        });
        assert!(called);
    }

    #[test]
    fn test_conditional_call_elidable() {
        let result = conditional_call_elidable(Some(42), || 99);
        assert_eq!(result, 42);
        let result = conditional_call_elidable(None, || 99);
        assert_eq!(result, 99);
    }

    #[test]
    fn test_record_exact_value_passthrough() {
        assert_eq!(record_exact_value(42, 42), 42);
    }

    #[test]
    fn test_loop_unrolling_heuristic() {
        let lst: &[i32] = &[];
        assert!(loop_unrolling_heuristic(lst, 0, 2));
        let lst = &[1, 2];
        assert!(loop_unrolling_heuristic(lst, 2, 2));
        let lst = &[1, 2, 3];
        assert!(!loop_unrolling_heuristic(lst, 3, 2));
    }

    #[test]
    fn test_enable_opts_all() {
        assert_eq!(PARAMETERS.enable_opts, EnableOpts::All);
        assert_eq!(
            ENABLE_ALL_OPTS,
            "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll"
        );
    }
}
