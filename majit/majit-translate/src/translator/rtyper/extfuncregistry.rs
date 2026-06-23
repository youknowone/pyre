//! Port of `rpython/rtyper/extfuncregistry.py`.

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

pub fn register_external_functions() -> Result<&'static [ExtFuncEntry], TyperError> {
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
