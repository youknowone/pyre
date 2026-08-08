//! Test-only smoke coverage for the `#[pyre_class]` / `#[pyre_methods]` /
//! `#[pyre_function]` / `py_module!` macro arms.
//!
//! These probes previously lived in the `_random` module proper, where
//! they leaked non-PyPy public API (`_unwrap_alias_probe`, `_PROBE_CONST`,
//! `_ProbeError`, `__reduce__`, `raw_state`, …).  Relocated here under
//! `#[cfg(test)]` so the macro codegen stays exercised end-to-end while
//! `_random`'s import-time surface matches `pypy/module/_random` exactly.
//!
//! Compiling the test build forces every macro arm below to expand, which
//! is the codegen smoke check; the `#[test]` functions additionally run
//! the wrappers that need no module-init bootstrap.

use pyre_object::*;

/// `#[pyre_class]` typed payload exercising getter/setter/deleter,
/// `__reduce__`, and the declarative `base = <expr>` arm.
#[crate::pyre_class("_pyre_smoke.Demo")]
#[derive(Default)]
pub struct Demo {
    pub state: u64,
}

#[crate::pyre_methods(
    doc = "Demo() -> smoke-test typed payload.",
    weakrefable,
    // `base = <expr>` arm — `object` is the implicit default, so this is
    // behaviorally identical while exercising the declarative-base plumbing.
    base = crate::typedef::w_object()
)]
impl Demo {
    fn __init__(&mut self, #[default(0i64)] seed: i64) {
        self.state = seed as u64;
    }
    fn getstate(&self) -> PyObjectRef {
        crate::pytuple![self.state as i64]
    }
    // `__reduce__` pickling-hook arm.
    fn __reduce__(&self) -> PyObjectRef {
        crate::pytuple![type_object(), crate::pytuple![], self.getstate()]
    }
    // A positional-or-keyword parameter plus a keyword-only default: the
    // instance-method arm must build a `Signature` (`self` posonly, then
    // `factor`, then a `marker_kwonly()` tail with `bias`) so the call path
    // binds keywords by name and the wrapper preamble needs no marker-dict
    // strip.
    fn combine(
        &self,
        factor: i64,
        #[kwonly]
        #[default(0i64)]
        bias: i64,
    ) -> i64 {
        self.state as i64 * factor + bias
    }
    // `#[getter]` / `#[setter]` / `#[deleter]` GetSetProperty quad.
    #[getter(doc = "raw 64-bit state as a signed int")]
    fn raw_state(&self) -> i64 {
        self.state as i64
    }
    #[setter]
    fn set_raw_state(&mut self, v: i64) {
        self.state = v as u64;
    }
    #[deleter("raw_state")]
    fn del_raw_state(&mut self) {
        self.state = 0;
    }
}

/// `PyPath` typed-receiver alias.
#[crate::pyre_function]
fn _seed_from_path(path: PyPath) -> i64 {
    path.into_iter().map(|b| b as i64).sum()
}

/// `Vec<i64>` auto return-wrap.
#[crate::pyre_function]
fn _path_bytes(path: PyPath) -> Vec<i64> {
    path.into_iter().map(|b| b as i64).collect()
}

/// One parameter per text / int unwrap alias, so the generated unwrap +
/// binding-type expansion is exercised for each.
#[crate::pyre_function]
fn _unwrap_alias_probe(
    u: PyUnicode,
    u8s: PyUtf8,
    ton: PyTextOrNone,
    t0n: PyText0OrNone,
    buf: PyBufferStr,
    cnn: PyCNonNegInt,
) -> i64 {
    let mut acc = u.len() as i64 + u8s.len() as i64 + buf.len() as i64 + cnn as i64;
    acc += ton.map(|s| s.len() as i64).unwrap_or(-1);
    acc += t0n.map(|s| s.len() as i64).unwrap_or(-1);
    acc
}

/// A Signature-bound scope includes keyword-only slots in its flat argument
/// array.  The typed wrapper must not recount those slots as positionals.
#[crate::pyre_function]
fn _kwonly_bound_probe(
    value: i64,
    #[kwonly]
    #[default(0i64)]
    adjustment: i64,
) -> i64 {
    value + adjustment
}

/// A `#[posonly]` marker on the first non-positional-only parameter ends the
/// positional-only run before it: `base` is positional-only, so the derived
/// `Signature` carries `posonlyargcount == 1` and a keyword named `base` is
/// rejected by `raise_if_posonly_kwds`.
#[crate::pyre_function]
fn _posonly_bound_probe(
    base: i64,
    #[posonly]
    #[default(0i64)]
    offset: i64,
) -> i64 {
    base + offset
}

crate::py_module! {
    "_pyre_smoke",
    interpleveldefs: {
        "Demo" => type_object(),
        "_seed_from_path" => crate::make_builtin_function("_seed_from_path", _seed_from_path),
        "_path_bytes" => crate::make_builtin_function("_path_bytes", _path_bytes),
        "_unwrap_alias_probe" =>
            crate::make_builtin_function("_unwrap_alias_probe", _unwrap_alias_probe),
    },
    int_constants: {
        // `int_constants:` arm — a plain integer module constant.
        "_PROBE_CONST" => 42,
    },
    exceptions: {
        // `exceptions:` arm — a module-local exception class.
        "_ProbeError" => crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before smoke init"),
    },
    appleveldefs: {
        // `appleveldefs:` arm — a pure-Python helper sharing the namespace.
        "app_smoke.py" => ["_ascii_seed"],
    },
    inline_app: {
        // `inline_app:` arm — an inline Python snippet.
        "def _is_even(n):\n    return n % 2 == 0\n" => ["_is_even"],
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `#[pyre_function]` unwrap-alias wrappers convert their typed
    /// arguments and return a wrapped int.  Exercises `PyUnicode` /
    /// `PyUtf8` / `PyTextOrNone` / `PyText0OrNone` / `PyBufferStr` /
    /// `PyCNonNegInt` through the space-level converters.
    #[test]
    fn unwrap_alias_probe_runs() {
        crate::typedef::init_typeobjects();
        let args = [
            w_str_new("ab"),
            w_str_new("cde"),
            w_none(),
            w_str_new("fghi"),
            pyre_object::bytesobject::w_bytes_from_bytes(b"jk"),
            w_int_new(7),
        ];
        let result = _unwrap_alias_probe(&args).expect("probe should succeed");
        // 2 + 3 + 2 + 7 (u + u8s + buf + cnn) - 1 (ton None) + 4 (t0n len)
        assert_eq!(unsafe { w_int_get_value(result) }, 2 + 3 + 2 + 7 - 1 + 4);
    }

    /// A missing required argument raises `TypeError` instead of panicking.
    #[test]
    fn missing_required_arg_is_type_error() {
        crate::typedef::init_typeobjects();
        let err = _seed_from_path(&[]).expect_err("missing path should error");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
    }

    #[test]
    fn keyword_only_bound_slot_is_not_counted_as_positional() {
        crate::typedef::init_typeobjects();
        let signature = _kwonly_bound_probe_pyre_sig().expect("derived signature");
        let bound = crate::call::bind_kwargs_to_signature(
            &signature,
            "_kwonly_bound_probe",
            &[w_int_new(40)],
            &[(rustpython_wtf8::Wtf8Buf::from("adjustment"), w_int_new(2))],
        )
        .expect("signature binding");
        let result = _kwonly_bound_probe(&bound).expect("bound keyword-only scope");
        assert_eq!(unsafe { w_int_get_value(result) }, 42);
    }

    #[test]
    fn posonly_marker_makes_leading_param_positional_only() {
        crate::typedef::init_typeobjects();
        let signature = _posonly_bound_probe_pyre_sig().expect("derived signature");
        assert_eq!(signature.posonlyargcount, 1);
        assert_eq!(signature.argnames, vec!["base", "offset"]);

        // The positional-only `base` binds fine by position, and `offset`
        // still binds by keyword.
        let bound = crate::call::bind_kwargs_to_signature(
            &signature,
            "_posonly_bound_probe",
            &[w_int_new(40)],
            &[(rustpython_wtf8::Wtf8Buf::from("offset"), w_int_new(2))],
        )
        .expect("signature binding");
        let result = _posonly_bound_probe(&bound).expect("bound positional-only scope");
        assert_eq!(unsafe { w_int_get_value(result) }, 42);

        // Passing the positional-only `base` as a keyword is a TypeError.
        let err = crate::call::bind_kwargs_to_signature(
            &signature,
            "_posonly_bound_probe",
            &[],
            &[(rustpython_wtf8::Wtf8Buf::from("base"), w_int_new(40))],
        )
        .expect_err("positional-only name as keyword must error");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
    }

    /// `Vec<i64>` return auto-wraps to a list.
    #[test]
    fn path_bytes_returns_list() {
        crate::typedef::init_typeobjects();
        let result = _path_bytes(&[w_str_new("AB")]).expect("path bytes");
        assert!(unsafe { is_list(result) });
        assert_eq!(unsafe { w_list_len(result) }, 2);
    }

    /// PyPy's `generic_new_descr` accepts the constructor arguments and
    /// leaves them for `__init__`; the synthesized allocator must do the same.
    #[test]
    fn synthesized_new_accepts_init_arguments() {
        crate::typedef::init_typeobjects();
        let cls = type_object();
        let seed = w_int_new(37);
        let obj = __pyre_wrap___new__(&[cls, seed]).expect("synthesized __new__");
        let demo = Demo::from_obj(obj).expect("Demo allocation");
        assert_eq!(demo.state, 0);
        __pyre_wrap___init__(&[obj, seed]).expect("Demo.__init__");
        assert_eq!(Demo::from_obj(obj).expect("initialized Demo").state, 37);
    }

    /// A `#[pyre_methods]` instance method with a positional-or-keyword and a
    /// keyword-only parameter binds identically whether the keyword arrives
    /// positionally or by name through `bind_kwargs_to_signature` — the same
    /// invariant the caller relies on to hand the wrapper a marker-free,
    /// PY_NULL-padded scope.
    #[test]
    fn instance_method_binds_keyword_only_through_signature() {
        crate::typedef::init_typeobjects();
        let cls = type_object();
        let obj = __pyre_wrap___new__(&[cls]).expect("synthesized __new__");
        __pyre_wrap___init__(&[obj, w_int_new(7)]).expect("Demo.__init__");

        // The `Signature` the instance-method arm derives for
        // `combine(&self, factor, #[kwonly] bias)`: `self` positional-only,
        // then `factor`, then a keyword-only `bias`.
        let signature = crate::gateway::Signature::new(
            vec!["self", "factor", "bias"],
            None,
            None,
            /*kwonlyargcount*/ 1,
            /*posonlyargcount*/ 1,
        );
        let bound = crate::call::bind_kwargs_to_signature(
            &signature,
            "combine",
            &[obj, w_int_new(3)],
            &[(rustpython_wtf8::Wtf8Buf::from("bias"), w_int_new(5))],
        )
        .expect("signature binding");
        // 7 * 3 + 5 through the bound (marker-free, PY_NULL-padded) scope.
        let via_keyword = __pyre_wrap_combine(&bound).expect("keyword-bound combine");
        assert_eq!(unsafe { w_int_get_value(via_keyword) }, 26);

        // The all-positional call omits `bias`, so its `#[default(0)]` applies.
        let via_positional = __pyre_wrap_combine(&[obj, w_int_new(3)]).expect("positional combine");
        assert_eq!(unsafe { w_int_get_value(via_positional) }, 21);
    }

    /// TypeDef ownership is process-global: another OS thread must observe
    /// the exact same Python type object, not a fresh TLS allocation.
    #[test]
    fn generated_type_object_is_process_global() {
        crate::typedef::init_typeobjects();
        let expected = type_object() as usize;
        let observed = std::thread::spawn(|| type_object() as usize)
            .join()
            .expect("type lookup thread");
        assert_eq!(observed, expected);
    }
}
