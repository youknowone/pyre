//! Flow-space special cases and builtin objects.
//!
//! RPython basis: `rpython/flowspace/specialcase.py`.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::model::{BuiltinFunction, BuiltinObject, BuiltinType, ConstValue, ExceptionClass};

static BUILTINS: LazyLock<HashMap<String, ConstValue>> = LazyLock::new(|| {
    let mut builtins = HashMap::new();
    builtins.insert(
        "print".to_owned(),
        ConstValue::Builtin(BuiltinObject::Function(BuiltinFunction::Print)),
    );
    builtins.insert(
        "getattr".to_owned(),
        ConstValue::Builtin(BuiltinObject::Function(BuiltinFunction::GetAttr)),
    );
    builtins.insert(
        "__import__".to_owned(),
        ConstValue::Builtin(BuiltinObject::Function(BuiltinFunction::Import)),
    );
    builtins.insert(
        "locals".to_owned(),
        ConstValue::Builtin(BuiltinObject::Function(BuiltinFunction::Locals)),
    );
    builtins.insert(
        "all".to_owned(),
        ConstValue::Builtin(BuiltinObject::Function(BuiltinFunction::All)),
    );
    builtins.insert(
        "any".to_owned(),
        ConstValue::Builtin(BuiltinObject::Function(BuiltinFunction::Any)),
    );
    builtins.insert(
        "tuple".to_owned(),
        ConstValue::Builtin(BuiltinObject::Type(BuiltinType::Tuple)),
    );
    builtins.insert(
        "list".to_owned(),
        ConstValue::Builtin(BuiltinObject::Type(BuiltinType::List)),
    );
    builtins.insert(
        "set".to_owned(),
        ConstValue::Builtin(BuiltinObject::Type(BuiltinType::Set)),
    );
    builtins.insert(
        "type".to_owned(),
        ConstValue::Builtin(BuiltinObject::Type(BuiltinType::Type)),
    );
    for name in [
        "AssertionError",
        "BaseException",
        "Exception",
        "ImportError",
        "NotImplementedError",
        "RuntimeError",
        "StackOverflow",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
        "_StackOverflow",
    ] {
        builtins.insert(
            name.to_owned(),
            ConstValue::ExceptionClass(ExceptionClass::builtin(name)),
        );
    }
    builtins
});

pub fn lookup_builtin(name: &str) -> Option<ConstValue> {
    BUILTINS.get(name).cloned()
}

pub fn builtin_function(function: BuiltinFunction) -> ConstValue {
    ConstValue::Builtin(BuiltinObject::Function(function))
}
