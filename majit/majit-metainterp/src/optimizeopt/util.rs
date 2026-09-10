//! RPython parity module for `rpython/jit/metainterp/optimizeopt/util.py`.
//!
//! PyPy's dispatcher builders are compile-time `make_dispatcher_method`
//! helpers. In Rust those dispatch tables are static `match` / method calls in
//! each optimization pass, so this module exposes the data-structure helpers
//! that remain meaningful at the Rust type level.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use majit_gc::shadow_stack::OwnerRootGuard;
use majit_ir::Value;
use majit_ir::operand::Operand;
use std::cell::RefCell;
use std::rc::Rc;

/// `history.py ConstPtr.value` is traced storage. Native cached constants
/// retain their own collector-updated slot instead of copying an address.
/// The guard never retains the dictionary, so its final owner retires roots.
enum CachedConst {
    Scalar(Value),
    Ref(OwnerRootGuard),
}

impl CachedConst {
    fn new(value: Value) -> Self {
        match value {
            Value::Ref(root) => Self::Ref(OwnerRootGuard::new(root)),
            value => Self::Scalar(value),
        }
    }

    fn value(&self) -> Value {
        match self {
            Self::Scalar(value) => *value,
            Self::Ref(root) => Value::Ref(root.get()),
        }
    }
}

struct ArgsKey(Vec<CachedConst>);

impl ArgsKey {
    fn new(args: impl IntoIterator<Item = Value>) -> Self {
        // Root every argument before hashing any: identityhash may reserve an
        // old-generation shadow for a nursery object (minimark.find_shadow).
        Self(args.into_iter().map(CachedConst::new).collect())
    }
}

impl PartialEq for ArgsKey {
    fn eq(&self, other: &Self) -> bool {
        // util.py args_eq / history.py Const.same_box: typed value equality,
        // including bitwise Float equality and CURRENT Ref pointer identity.
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|(a, b)| a.value() == b.value())
    }
}

impl Eq for ArgsKey {}

impl Hash for ArgsKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Native Hash encoding of util.py args_hash (hash numbers themselves
        // are not a wire format). Ref hash must be stable across forwarding, like
        // history.py ConstPtr._get_hash_; raw Value::hash hashes its address.
        let mut hash = 0x345678_u64;
        for arg in &self.0 {
            let word = match arg.value() {
                Value::Int(value) => value as u64,
                Value::Float(value) => value.to_bits(),
                Value::Ref(root) => majit_gc::gc_id_or_identityhash(root.0) as u64,
                Value::Void => 17,
            };
            hash = hash.wrapping_mul(1000003) ^ word;
        }
        hash.hash(state);
    }
}

/// `util.py args_dict`, specialized to CALL_PURE's constant argument lists.
/// `compile.py` and `optimizer.py` share the metainterpreter's dictionary,
/// not independent snapshots. Clone retains that identity and its roots.
/// Collection updates only the independently owned Const slots, never this
/// map through a borrowed TraceCtx/Optimizer. No GC-time clear or rehash.
#[derive(Clone, Default)]
pub struct ArgsDict(RefCell<Option<Rc<RefCell<indexmap::IndexMap<ArgsKey, CachedConst>>>>>);

impl std::fmt::Debug for ArgsDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArgsDict")
            .field(
                "len",
                &self
                    .0
                    .borrow()
                    .as_ref()
                    .map(|m| m.borrow().len())
                    .unwrap_or(0),
            )
            .finish()
    }
}

pub fn args_dict() -> ArgsDict {
    ArgsDict(RefCell::new(Some(Rc::new(RefCell::new(
        indexmap::IndexMap::new(),
    )))))
}

impl ArgsDict {
    pub fn insert(&self, args: Vec<Value>, value: Value) {
        let key = ArgsKey::new(args);
        let value = CachedConst::new(value);
        let mut slot = self.0.borrow_mut();
        if slot.is_none() {
            *slot = Some(Rc::new(RefCell::new(indexmap::IndexMap::new())));
        }
        slot.as_ref()
            .expect("args_dict insert")
            .borrow_mut()
            .insert(key, value);
    }

    pub fn get(&self, args: &[Value]) -> Option<Value> {
        let key = ArgsKey::new(args.iter().copied());
        self.0
            .borrow()
            .as_ref()
            .and_then(|m| m.borrow().get(&key).map(CachedConst::value))
    }
}

/// util.py `args_eq`.
///
/// Uses `same_box`, so constants compare by value while regular boxes compare
/// by object identity.
pub fn args_eq(args1: &[Option<Operand>], args2: &[Option<Operand>]) -> bool {
    args1.len() == args2.len()
        && args1
            .iter()
            .zip(args2)
            .all(|(arg1, arg2)| match (arg1, arg2) {
                (None, None) => true,
                (Some(arg1), Some(arg2)) => arg1.same_box(arg2),
                _ => false,
            })
}

/// util.py `args_hash`.
///
/// The exact Python integer hash width is not part of Rust's hash API; this
/// preserves the load-bearing contract with `args_eq`: equal argument lists
/// produce equal hashes, with constants hashed by value and other boxes by
/// identity.
pub fn args_hash(args: &[Option<Operand>]) -> u64 {
    let mut state = DefaultHasher::new();
    0x345678_u64.hash(&mut state);
    for arg in args {
        match arg {
            None => 17_u8.hash(&mut state),
            Some(arg) => hash_arg(arg, &mut state),
        }
    }
    state.finish()
}

fn hash_arg<H: Hasher>(arg: &Operand, state: &mut H) {
    if let Some(value) = arg.const_value() {
        value.hash(state);
    } else {
        arg.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::{args_eq, args_hash};
    use majit_ir::OpRef;
    use majit_ir::operand::Operand;
    use majit_ir::value::Const;

    #[test]
    fn args_eq_uses_same_box_const_value_semantics() {
        let a = Some(Operand::const_(Const::Int(5)));
        let b = Some(Operand::const_(Const::Int(5)));
        assert!(args_eq(std::slice::from_ref(&a), std::slice::from_ref(&b)));
        assert_eq!(args_hash(&[a]), args_hash(&[b]));
    }

    #[test]
    fn args_eq_distinguishes_non_const_box_identity() {
        let a = Some(Operand::bound_from_opref(OpRef::int_op(0)));
        let b = Some(Operand::bound_from_opref(OpRef::int_op(0)));
        assert!(!args_eq(&[a], &[b]));
    }

    #[test]
    fn args_dict_isolates_fresh_caches_and_uses_typed_constant_equality() {
        use majit_ir::Value;
        let dict = super::args_dict();
        dict.insert(vec![Value::Float(0.0)], Value::Int(1));
        dict.insert(vec![Value::Float(-0.0)], Value::Int(2));
        dict.insert(vec![Value::Int(0)], Value::Int(3));
        let nan = f64::from_bits(0x7ff8000000000001);
        dict.insert(vec![Value::Float(nan)], Value::Int(4));
        assert_eq!(dict.get(&[Value::Float(0.0)]), Some(Value::Int(1)));
        assert_eq!(dict.get(&[Value::Float(-0.0)]), Some(Value::Int(2)));
        assert_eq!(dict.get(&[Value::Int(0)]), Some(Value::Int(3)));
        let fresh = super::args_dict();
        assert_eq!(fresh.get(&[Value::Int(0)]), None);
        fresh.insert(vec![Value::Int(0)], Value::Int(5));
        assert_eq!(dict.get(&[Value::Int(0)]), Some(Value::Int(3)));
        assert_eq!(dict.get(&[Value::Float(nan)]), Some(Value::Int(4)));
        assert_eq!(
            dict.get(&[Value::Float(f64::from_bits(nan.to_bits() + 1))]),
            None
        );
    }
}
