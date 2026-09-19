//! Port of `rpython/rtyper/extfuncregistry.py`.
//!
//! Deferred port: upstream additionally imports `rposix`, `rposix_stat`,
//! `rposix_environ`, and `rtime` (`extfuncregistry.py`) for their
//! `register_external` side effects, registering the os/stat/environ/time
//! external functions. Those `rlib` modules are not ported here yet, so
//! only the math/rfloat registrations below are present. Converge by
//! porting `rposix`/`rposix_stat`/`rposix_environ`/`rtime` and adding
//! their `register_external` entries to `register_external_functions`.

use std::sync::OnceLock;

use crate::flowspace::model::HostObject;

use super::error::TyperError;
use super::extfunc::{self, ExtFuncEntry, ExternalAnnotation};

fn math_function(name: &str) -> HostObject {
    HostObject::new_builtin_callable(format!("math.{name}"))
}

fn rfloat_function(name: &str) -> HostObject {
    HostObject::new_builtin_callable(format!("rpython.rlib.rfloat.{name}"))
}

/// RPython `_register` from `extfuncregistry.py`.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub const _REGISTER: &[(&str, &[(&str, &[&str], &str)])] = &[
    ("rpython.rlib.rfloat", &[("isfinite", &["float"], "bool")]),
    (
        "math",
        &[
            ("copysign", &["float", "float"], "float"),
            ("isinf", &["float"], "bool"),
            ("isnan", &["float"], "bool"),
            ("floor", &["float"], "float"),
            ("sqrt", &["float"], "float"),
            ("log", &["float"], "float"),
            ("log10", &["float"], "float"),
            ("log1p", &["float"], "float"),
            ("sin", &["float"], "float"),
            ("cos", &["float"], "float"),
            ("atan2", &["float", "float"], "float"),
            ("hypot", &["float", "float"], "float"),
            ("frexp", &["float"], "(float, int)"),
            ("ldexp", &["float", "int"], "float"),
            ("modf", &["float"], "(float, float)"),
            ("fmod", &["float", "float"], "float"),
            ("pow", &["float", "float"], "float"),
        ],
    ),
];

static REGISTERED_EXTERNALS: OnceLock<Result<Vec<ExtFuncEntry>, TyperError>> = OnceLock::new();

pub(crate) fn register_external_functions() -> Result<&'static [ExtFuncEntry], TyperError> {
    let result = REGISTERED_EXTERNALS.get_or_init(|| {
        let mut entries = Vec::new();
        for name in super::lltypesystem::module::ll_math::UNARY_MATH_FUNCTIONS {
            entries.push(extfunc::register_external(
                math_function(name),
                vec![ExternalAnnotation::Float],
                Some(ExternalAnnotation::Float),
                Some(format!("ll_math.ll_math_{name}")),
                None,
                None,
                true,
            )?);
        }
        for (module, methods) in _REGISTER {
            for (name, arg_types, return_type) in *methods {
                let function = if *module == "math" {
                    math_function(name)
                } else {
                    rfloat_function(name)
                };
                entries.push(extfunc::register_external(
                    function,
                    arg_types
                        .iter()
                        .map(|arg| annotation_by_name(arg))
                        .collect(),
                    Some(annotation_by_name(return_type)),
                    Some(format!("ll_math.ll_math_{name}")),
                    None,
                    None,
                    true,
                )?);
            }
        }
        // The front emits the C llexternal crate path (`ll_math::math_floor`),
        // not the Python `math.floor` host the loops above key on.  Register
        // that spelling from the same table the front reads so the two
        // naming worlds meet.  Result is always `float`: that is the C
        // llexternal / `register_external(..., [float], float)` annotation,
        // even when `ll_math_ceil` (the wrapper) returns `Result`.
        for row in super::lltypesystem::module::ll_math::F64_METHOD_LLEXTERNALS {
            let args = vec![ExternalAnnotation::Float; row.arity];
            entries.push(extfunc::register_external(
                HostObject::new_builtin_callable(format!("ll_math.{}", row.name)),
                args,
                Some(ExternalAnnotation::Float),
                Some(format!("ll_math.{}", row.name)),
                None,
                None,
                true,
            )?);
        }
        Ok(entries)
    });
    match result {
        Ok(entries) => Ok(entries),
        Err(err) => Err(err.clone()),
    }
}

fn annotation_by_name(name: &str) -> ExternalAnnotation {
    match name {
        "float" => ExternalAnnotation::Float,
        "int" => ExternalAnnotation::Int,
        "bool" => ExternalAnnotation::Bool,
        "(float, int)" => {
            ExternalAnnotation::Tuple(vec![ExternalAnnotation::Float, ExternalAnnotation::Int])
        }
        "(float, float)" => {
            ExternalAnnotation::Tuple(vec![ExternalAnnotation::Float, ExternalAnnotation::Float])
        }
        other => panic!("unsupported extfuncregistry annotation {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_ll_math_external_names() {
        let entries = register_external_functions().expect("register extfuncregistry");
        assert!(
            entries
                .iter()
                .any(|entry| entry.name == "ll_math.ll_math_sqrt")
        );
        assert!(
            entries
                .iter()
                .any(|entry| entry.name == "ll_math.ll_math_isfinite")
        );
    }

    #[test]
    fn registers_f64_method_llexternal_crate_paths() {
        let entries = register_external_functions().expect("register extfuncregistry");
        for row in super::super::lltypesystem::module::ll_math::F64_METHOD_LLEXTERNALS {
            let entry = entries
                .iter()
                .find(|entry| entry.name == format!("ll_math.{}", row.name))
                .unwrap_or_else(|| panic!("missing crate-path external {}", row.name));
            assert_eq!(entry.signature_args.len(), row.arity, "{}", row.name);
            assert!(
                entry
                    .signature_args
                    .iter()
                    .all(|arg| *arg == ExternalAnnotation::Float),
                "{}",
                row.name
            );
            assert_eq!(
                entry.signature_result,
                ExternalAnnotation::Float,
                "{}",
                row.name
            );
        }
        let floor = entries
            .iter()
            .find(|entry| entry.name == "ll_math.math_floor")
            .expect("math_floor");
        let ceil = entries
            .iter()
            .find(|entry| entry.name == "ll_math.math_ceil")
            .expect("math_ceil");
        assert_eq!(floor.signature_args, vec![ExternalAnnotation::Float]);
        assert_eq!(ceil.signature_args, vec![ExternalAnnotation::Float]);
        let hypot = entries
            .iter()
            .find(|entry| entry.name == "ll_math.math_hypot")
            .expect("math_hypot");
        assert_eq!(
            hypot.signature_args,
            vec![ExternalAnnotation::Float, ExternalAnnotation::Float]
        );
    }

    #[test]
    fn frexp_and_modf_keep_tuple_result_shapes() {
        let entries = register_external_functions().expect("register extfuncregistry");
        let frexp = entries
            .iter()
            .find(|entry| entry.name == "ll_math.ll_math_frexp")
            .expect("frexp");
        let modf = entries
            .iter()
            .find(|entry| entry.name == "ll_math.ll_math_modf")
            .expect("modf");
        assert!(matches!(
            frexp.signature_result,
            ExternalAnnotation::Tuple(_)
        ));
        assert!(matches!(
            modf.signature_result,
            ExternalAnnotation::Tuple(_)
        ));
    }
}
