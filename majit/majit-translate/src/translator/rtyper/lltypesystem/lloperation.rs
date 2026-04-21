//! RPython `rpython/rtyper/lltypesystem/lloperation.py`.
//!
//! This port currently carries the `LLOp` descriptor table plus
//! `enum_ops_without_sideeffects()`, which is the surface consumed by
//! `translator/simplify.py`.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LLOp {
    pub opname: &'static str,
    pub sideeffects: bool,
    pub canfold: bool,
    pub tryfold: bool,
    pub canraise: Vec<&'static str>,
    pub canmallocgc: bool,
    pub canrun: bool,
    pub revdb_protect: bool,
}

impl LLOp {
    pub fn new(
        sideeffects: bool,
        canfold: bool,
        canraise: &[&'static str],
        canmallocgc: bool,
        canrun: bool,
        tryfold: bool,
        revdb_protect: bool,
    ) -> Self {
        let mut sideeffects = sideeffects;
        if canfold {
            sideeffects = false;
        }
        let mut canraise_vec = canraise.to_vec();
        if canmallocgc
            && !canraise_vec.contains(&"MemoryError")
            && !canraise_vec.contains(&"Exception")
        {
            canraise_vec.push("MemoryError");
        }
        LLOp {
            opname: "?",
            sideeffects,
            canfold,
            tryfold: tryfold || canfold,
            canraise: canraise_vec,
            canmallocgc,
            canrun: canrun || canfold,
            revdb_protect,
        }
    }
}

fn insert_llop(ops: &mut HashMap<&'static str, LLOp>, opname: &'static str, mut opdesc: LLOp) {
    opdesc.opname = opname;
    ops.insert(opname, opdesc);
}

pub fn ll_operations() -> &'static HashMap<&'static str, LLOp> {
    static LL_OPERATIONS: OnceLock<HashMap<&'static str, LLOp>> = OnceLock::new();
    LL_OPERATIONS.get_or_init(|| {
        let mut ops = HashMap::new();

        insert_llop(
            &mut ops,
            "direct_call",
            LLOp::new(true, false, &["Exception"], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "indirect_call",
            LLOp::new(true, false, &["Exception"], false, false, false, false),
        );

        // __________ numeric operations __________

        insert_llop(
            &mut ops,
            "bool_not",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "char_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "char_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "char_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "char_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "char_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "char_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "unichar_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "unichar_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "int_is_true",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_neg",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_abs",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_invert",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "int_add",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_sub",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_mul",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_floordiv",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_mod",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_and",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_or",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_lshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_rshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_xor",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_between",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_force_ge_zero",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "int_add_ovf",
            LLOp::new(true, false, &["OverflowError"], false, false, true, false),
        );
        insert_llop(
            &mut ops,
            "int_add_nonneg_ovf",
            LLOp::new(true, false, &["OverflowError"], false, false, true, false),
        );
        insert_llop(
            &mut ops,
            "int_sub_ovf",
            LLOp::new(true, false, &["OverflowError"], false, false, true, false),
        );
        insert_llop(
            &mut ops,
            "int_mul_ovf",
            LLOp::new(true, false, &["OverflowError"], false, false, true, false),
        );

        insert_llop(
            &mut ops,
            "uint_is_true",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_invert",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_add",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_sub",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_mul",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_floordiv",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_mod",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_and",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_or",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_lshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_rshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "uint_xor",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "float_is_true",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_neg",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_abs",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_add",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_sub",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_mul",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_truediv",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "float_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "llong_is_true",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_neg",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_abs",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_invert",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_add",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_sub",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_mul",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_floordiv",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_mod",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_and",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_or",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_lshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_rshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "llong_xor",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "ullong_is_true",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_invert",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_add",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_sub",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_mul",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_floordiv",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_mod",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_and",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_or",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_lshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_rshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ullong_xor",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "lllong_is_true",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_neg",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_abs",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_invert",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_add",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_sub",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_mul",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_floordiv",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_mod",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_and",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_or",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_lshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_rshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "lllong_xor",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "ulllong_is_true",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_invert",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_add",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_sub",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_mul",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_floordiv",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_mod",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_and",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_or",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_lshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_rshift",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ulllong_xor",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "cast_primitive",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_bool_to_int",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_bool_to_uint",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_bool_to_float",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_char_to_int",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_unichar_to_int",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_int_to_char",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_int_to_unichar",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_int_to_uint",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_int_to_float",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_int_to_longlong",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_uint_to_int",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_uint_to_float",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_longlong_to_float",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_ulonglong_to_float",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_float_to_int",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_float_to_uint",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_float_to_longlong",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_float_to_ulonglong",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "truncate_longlong_to_int",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "force_cast",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "convert_float_bytes_to_longlong",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "convert_longlong_bytes_to_float",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "likely",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "unlikely",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        // __________ pointer operations __________

        insert_llop(
            &mut ops,
            "malloc",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "malloc_varsize",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "shrink_array",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "zero_gc_pointers_inside",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "free",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "getfield",
            LLOp::new(false, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "getarrayitem",
            LLOp::new(false, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "getarraysize",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "getsubstruct",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "getinteriorfield",
            LLOp::new(false, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "getinteriorarraysize",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "setinteriorfield",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "bare_setinteriorfield",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "getarraysubstruct",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "setfield",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "bare_setfield",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "setarrayitem",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "bare_setarrayitem",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_pointer",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ptr_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ptr_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ptr_nonzero",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ptr_iszero",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_ptr_to_int",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_int_to_ptr",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "direct_fieldptr",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "direct_arrayitems",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "direct_ptradd",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_opaque_ptr",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "length_of_simple_gcarray_from_opaque",
            LLOp::new(false, false, &[], false, false, false, false),
        );

        // __________ address operations __________

        insert_llop(
            &mut ops,
            "boehm_malloc",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "boehm_malloc_atomic",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "boehm_register_finalizer",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "boehm_disappearing_link",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "raw_malloc",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "raw_malloc_usage",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "raw_free",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "raw_memclear",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "raw_memset",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "raw_memcopy",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "raw_memmove",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "raw_load",
            LLOp::new(false, false, &[], false, true, false, true),
        );
        insert_llop(
            &mut ops,
            "raw_store",
            LLOp::new(true, false, &[], false, true, false, true),
        );
        insert_llop(
            &mut ops,
            "bare_raw_store",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "gc_load_indexed",
            LLOp::new(false, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_store",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_store_indexed",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "track_alloc_start",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "track_alloc_stop",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_add",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_sub",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_delta",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_lt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_le",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_eq",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_ne",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_gt",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "adr_ge",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_ptr_to_adr",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_adr_to_ptr",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_adr_to_int",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_int_to_adr",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        insert_llop(
            &mut ops,
            "get_group_member",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "get_next_group_member",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "is_group_member_nonzero",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "extract_ushort",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "combine_ushort",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_gettypeptr_group",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "get_member_index",
            LLOp::new(true, true, &[], false, false, false, false),
        );

        // __________ used by the JIT ________

        insert_llop(
            &mut ops,
            "jit_marker",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_force_virtualizable",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_force_virtual",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_is_virtual",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_force_quasi_immutable",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_record_exact_class",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_record_exact_value",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_ffi_save_result",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_conditional_call",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_conditional_call_value",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_enter_portal_frame",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "jit_leave_portal_frame",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "get_exception_addr",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "get_exc_value_addr",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "do_malloc_fixedsize",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "do_malloc_fixedsize_clear",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "do_malloc_varsize",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "do_malloc_varsize_clear",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "get_write_barrier_failing_case",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "get_write_barrier_from_array_failing_case",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_get_type_info_group",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "ll_read_timestamp",
            LLOp::new(true, false, &[], false, true, false, true),
        );
        insert_llop(
            &mut ops,
            "ll_get_timestamp_unit",
            LLOp::new(true, false, &[], false, true, false, true),
        );
        insert_llop(
            &mut ops,
            "jit_record_known_result",
            LLOp::new(true, false, &[], false, true, false, false),
        );

        // __________ GC operations __________

        insert_llop(
            &mut ops,
            "gc__collect",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc__collect_step",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc__enable",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc__disable",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc__isenabled",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_free",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_fetch_exception",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_restore_exception",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_call_rtti_destructor",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_deallocate",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_reload_possibly_moved",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_identityhash",
            LLOp::new(false, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_id",
            LLOp::new(false, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_obtain_free_space",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "gc_set_max_heap_size",
            LLOp::new(true, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "gc_can_move",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_thread_run",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_thread_start",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_thread_die",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_thread_before_fork",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_thread_after_fork",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_writebarrier",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_writebarrier_before_copy",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_writebarrier_before_move",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_heap_stats",
            LLOp::new(true, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_pin",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_unpin",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc__is_pinned",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_bit",
            LLOp::new(false, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_get_rpy_roots",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_get_rpy_referents",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_get_rpy_memory_usage",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_get_rpy_type_index",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_is_rpy_instance",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_dump_rpy_heap",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_typeids_z",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_typeids_list",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_gettypeid",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_gcflag_extra",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_add_memory_pressure",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_get_stats",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_fq_next_dead",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_fq_register",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_ignore_finalizer",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_rawrefcount_init",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_rawrefcount_create_link_pypy",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_rawrefcount_create_link_pyobj",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_rawrefcount_mark_deallocating",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_rawrefcount_from_obj",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_rawrefcount_to_obj",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_rawrefcount_next_dead",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_move_out_of_nursery",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_increase_root_stack_depth",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_push_roots",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_pop_roots",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_enter_roots_frame",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_leave_roots_frame",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_save_root",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_restore_root",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_adr_of_nursery_free",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_adr_of_nursery_top",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_adr_of_root_stack_base",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_adr_of_root_stack_top",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_modified_shadowstack",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "gc_stack_bottom",
            LLOp::new(true, false, &[], false, true, false, false),
        );

        // __________ weakrefs __________

        insert_llop(
            &mut ops,
            "weakref_create",
            LLOp::new(false, false, &[], true, false, false, false),
        );
        insert_llop(
            &mut ops,
            "weakref_deref",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_ptr_to_weakrefptr",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "cast_weakrefptr_to_ptr",
            LLOp::new(false, false, &[], false, false, false, false),
        );

        // __________ misc operations __________

        insert_llop(
            &mut ops,
            "stack_current",
            LLOp::new(false, false, &[], false, false, false, true),
        );
        insert_llop(
            &mut ops,
            "keepalive",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "same_as",
            LLOp::new(true, true, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "hint",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "check_no_more_arg",
            LLOp::new(true, false, &["Exception"], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "check_self_nonzero",
            LLOp::new(true, false, &["Exception"], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "decode_arg",
            LLOp::new(true, false, &["Exception"], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "decode_arg_def",
            LLOp::new(true, false, &["Exception"], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "getslice",
            LLOp::new(true, false, &["Exception"], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "check_and_clear_exc",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "threadlocalref_addr",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "threadlocalref_get",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "threadlocalref_load",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "threadlocalref_store",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "threadlocalref_acquire",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "threadlocalref_release",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "threadlocalref_enum",
            LLOp::new(false, false, &[], false, false, false, false),
        );

        // __________ debugging __________

        insert_llop(
            &mut ops,
            "debug_view",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_print",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_start",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_stop",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "have_debug_prints",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "have_debug_prints_for",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_offset",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_flush",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_assert",
            LLOp::new(true, false, &[], false, false, true, false),
        );
        insert_llop(
            &mut ops,
            "debug_assert_not_none",
            LLOp::new(true, false, &[], false, false, true, false),
        );
        insert_llop(
            &mut ops,
            "debug_fatalerror",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_llinterpcall",
            LLOp::new(true, false, &["Exception"], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_start_traceback",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_record_traceback",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_catch_exception",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_reraise_traceback",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_print_traceback",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_nonnull_pointer",
            LLOp::new(true, false, &[], false, true, false, false),
        );
        insert_llop(
            &mut ops,
            "debug_forked",
            LLOp::new(true, false, &[], false, false, false, false),
        );

        // __________ instrumentation _________

        insert_llop(
            &mut ops,
            "instrument_count",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_stop_point",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_send_answer",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_breakpoint",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_get_value",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_get_unique_id",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_watch_save_state",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_watch_restore_state",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_weakref_create",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_weakref_deref",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_call_destructor",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_set_thread_breakpoint",
            LLOp::new(true, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_strtod",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_dtoa",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_modf",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_frexp",
            LLOp::new(false, false, &[], false, false, false, false),
        );
        insert_llop(
            &mut ops,
            "revdb_do_next_call",
            LLOp::new(true, false, &[], false, true, false, false),
        );

        ops
    })
}

pub fn enum_ops_without_sideeffects(raising_is_ok: bool) -> impl Iterator<Item = &'static str> {
    ll_operations().iter().filter_map(move |(opname, opdesc)| {
        if !opdesc.sideeffects && (opdesc.canraise.is_empty() || raising_is_ok) {
            Some(*opname)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canfold_forces_sideeffects_false() {
        assert!(!ll_operations().get("int_add").unwrap().sideeffects);
    }

    #[test]
    fn canmallocgc_adds_memoryerror_when_missing() {
        assert!(
            ll_operations()
                .get("malloc")
                .unwrap()
                .canraise
                .contains(&"MemoryError")
        );
    }

    #[test]
    fn enum_ops_without_sideeffects_filters_raising_ops_by_default() {
        let ops: Vec<_> = enum_ops_without_sideeffects(false).collect();
        assert!(ops.contains(&"int_add"));
        assert!(ops.contains(&"force_cast"));
        assert!(!ops.contains(&"check_no_more_arg"));
        assert!(!ops.contains(&"debug_assert"));
    }
}
