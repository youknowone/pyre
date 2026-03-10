/// Value types and constants for the JIT IR.
///
/// Translated from rpython/jit/metainterp/history.py.

/// The type of a value in the JIT IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// Machine-word signed integer (i64).
    Int,
    /// GC-managed reference (pointer).
    Ref,
    /// IEEE 754 double-precision float.
    Float,
    /// No value (void).
    Void,
}

impl Type {
    pub fn from_char(c: char) -> Self {
        match c {
            'i' => Type::Int,
            'r' | 'p' => Type::Ref,
            'f' => Type::Float,
            'v' | 'n' => Type::Void,
            _ => panic!("unknown type char: {c}"),
        }
    }

    pub fn to_char(self) -> char {
        match self {
            Type::Int => 'i',
            Type::Ref => 'r',
            Type::Float => 'f',
            Type::Void => 'v',
        }
    }
}

/// An opaque GC-managed reference.
///
/// In the actual runtime this wraps a pointer to a GC-managed object.
/// During tracing/optimization it may be a tagged value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GcRef(pub usize);

impl GcRef {
    pub const NULL: GcRef = GcRef(0);

    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// A concrete runtime value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Ref(GcRef),
    Void,
}

impl Value {
    pub fn get_type(&self) -> Type {
        match self {
            Value::Int(_) => Type::Int,
            Value::Float(_) => Type::Float,
            Value::Ref(_) => Type::Ref,
            Value::Void => Type::Void,
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Value::Int(v) => *v,
            _ => panic!("expected Int, got {:?}", self),
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Value::Float(v) => *v,
            _ => panic!("expected Float, got {:?}", self),
        }
    }

    pub fn as_ref(&self) -> GcRef {
        match self {
            Value::Ref(v) => *v,
            _ => panic!("expected Ref, got {:?}", self),
        }
    }
}

/// A constant value known at trace time.
///
/// Mirrors rpython/jit/metainterp/resoperation.py Const* classes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Const {
    Int(i64),
    Float(f64),
    Ref(GcRef),
}

impl Const {
    pub fn get_type(&self) -> Type {
        match self {
            Const::Int(_) => Type::Int,
            Const::Float(_) => Type::Float,
            Const::Ref(_) => Type::Ref,
        }
    }

    pub fn to_value(self) -> Value {
        match self {
            Const::Int(v) => Value::Int(v),
            Const::Float(v) => Value::Float(v),
            Const::Ref(v) => Value::Ref(v),
        }
    }
}

/// An input argument to a loop or bridge.
///
/// Mirrors rpython/jit/metainterp/resoperation.py InputArg* classes.
#[derive(Clone, Debug, PartialEq)]
pub struct InputArg {
    pub tp: Type,
    /// Index in the inputargs list.
    pub index: u32,
}

impl InputArg {
    pub fn new_int(index: u32) -> Self {
        InputArg {
            tp: Type::Int,
            index,
        }
    }

    pub fn new_ref(index: u32) -> Self {
        InputArg {
            tp: Type::Ref,
            index,
        }
    }

    pub fn new_float(index: u32) -> Self {
        InputArg {
            tp: Type::Float,
            index,
        }
    }

    pub fn from_type(tp: Type, index: u32) -> Self {
        InputArg { tp, index }
    }
}

/// Limit on the number of fail arguments per guard.
///
/// From history.py: FAILARGS_LIMIT = 1000
pub const FAILARGS_LIMIT: usize = 1000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_roundtrip() {
        for tp in [Type::Int, Type::Ref, Type::Float, Type::Void] {
            assert_eq!(Type::from_char(tp.to_char()), tp);
        }
    }

    #[test]
    fn test_value_types() {
        assert_eq!(Value::Int(42).get_type(), Type::Int);
        assert_eq!(Value::Float(3.14).get_type(), Type::Float);
        assert_eq!(Value::Ref(GcRef::NULL).get_type(), Type::Ref);
        assert_eq!(Value::Void.get_type(), Type::Void);
    }

    #[test]
    fn test_gcref_null() {
        assert!(GcRef::NULL.is_null());
        assert!(!GcRef(0x1234).is_null());
    }
}
