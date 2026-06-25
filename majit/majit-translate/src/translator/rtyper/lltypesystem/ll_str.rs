//! RPython `rpython/rtyper/lltypesystem/ll_str.py` parity module.
//!
//! Pyre currently materialises the ported `ll_int2hex` body as a helper
//! graph builder in [`super::rstr`], because callers need a `direct_call`
//! target rather than a host-side Rust string conversion.  Keep the
//! upstream file path available here while re-exporting only the helper
//! surface that is actually implemented.

use std::sync::LazyLock;

use crate::translator::rtyper::lltypesystem::lltype::{Array, LowLevelType};

pub use crate::translator::rtyper::lltypesystem::rstr::build_ll_int2hex_helper_graph;

/// RPython `CHAR_ARRAY = GcArray(Char)` (ll_str.py:5).
pub static CHAR_ARRAY: LazyLock<LowLevelType> =
    LazyLock::new(|| LowLevelType::Array(Box::new(Array::gc(LowLevelType::Char))));

/// RPython `ll_unsigned(i)` (ll_str.py:7-11) for Signed inputs.
pub fn ll_unsigned(i: i64) -> u64 {
    i as u64
}

fn unsigned_magnitude(i: i64) -> (bool, u64) {
    if i < 0 {
        (true, i.wrapping_neg() as u64)
    } else {
        (false, ll_unsigned(i))
    }
}

fn format_unsigned_digits(mut value: u64, radix: u64) -> String {
    if value == 0 {
        return "0".to_string();
    }
    let mut digits = Vec::new();
    while value != 0 {
        let digit = (value % radix) as u32;
        digits.push(std::char::from_digit(digit, radix as u32).expect("digit in radix"));
        value /= radix;
    }
    digits.iter().rev().collect()
}

/// RPython `@jit.elidable def ll_int2dec(val)` (ll_str.py:14-39).
pub fn ll_int2dec(val: i64) -> String {
    let (sign, value) = unsigned_magnitude(val);
    let digits = format_unsigned_digits(value, 10);
    if sign { format!("-{digits}") } else { digits }
}

/// RPython `@jit.elidable def ll_int2hex(i, addPrefix)` (ll_str.py:48-82).
pub fn ll_int2hex(i: i64, add_prefix: bool) -> String {
    let (sign, value) = unsigned_magnitude(i);
    let mut result = String::new();
    if sign {
        result.push('-');
    }
    if add_prefix {
        result.push_str("0x");
    }
    result.push_str(&format_unsigned_digits(value, 16));
    result
}

/// RPython `@jit.elidable def ll_int2oct(i, addPrefix)` (ll_str.py:85-119).
pub fn ll_int2oct(i: i64, add_prefix: bool) -> String {
    let (sign, value) = unsigned_magnitude(i);
    let mut result = String::new();
    if sign {
        result.push('-');
    }
    if add_prefix && value != 0 {
        result.push('0');
    }
    result.push_str(&format_unsigned_digits(value, 8));
    result
}

/// RPython `@jit.elidable def ll_int2bin(i, addPrefix)` (ll_str.py:122-156).
pub fn ll_int2bin(i: i64, add_prefix: bool) -> String {
    let (sign, value) = unsigned_magnitude(i);
    let mut result = String::new();
    if sign {
        result.push('-');
    }
    if add_prefix {
        result.push_str("0b");
    }
    result.push_str(&format_unsigned_digits(value, 2));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Hlvalue};
    use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, LowLevelValue};

    #[test]
    fn ll_int_format_helpers_match_ll_str_py_surface() {
        assert_eq!(ll_unsigned(-1), u64::MAX);
        assert_eq!(ll_int2dec(0), "0");
        assert_eq!(ll_int2dec(-12345), "-12345");
        assert_eq!(ll_int2hex(0, false), "0");
        assert_eq!(ll_int2hex(26, true), "0x1a");
        assert_eq!(ll_int2hex(-26, true), "-0x1a");
        assert_eq!(ll_int2oct(0, true), "0");
        assert_eq!(ll_int2oct(10, true), "012");
        assert_eq!(ll_int2oct(-10, true), "-012");
        assert_eq!(ll_int2bin(0, false), "0");
        assert_eq!(ll_int2bin(5, true), "0b101");
        assert_eq!(ll_int2bin(-5, true), "-0b101");
    }

    #[test]
    fn char_array_is_gc_array_of_char() {
        let LowLevelType::Array(arr) = &*CHAR_ARRAY else {
            panic!("CHAR_ARRAY must be an Array lowleveltype");
        };
        assert_eq!(arr.OF, LowLevelType::Char);
    }

    #[test]
    fn ll_str_path_exposes_ported_ll_int2hex_helper_graph() {
        let helper = build_ll_int2hex_helper_graph("ll_int2hex", true).expect("build ll_int2hex");
        let graph = helper.graph.borrow();

        let mut seen: Vec<*const std::cell::RefCell<crate::flowspace::model::Block>> = Vec::new();
        let mut queue = vec![graph.startblock.clone()];
        let mut opnames = Vec::new();
        let mut hex_chars: Option<ConstValue> = None;

        while let Some(block) = queue.pop() {
            let key = std::rc::Rc::as_ptr(&block);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);

            let b = block.borrow();
            for op in &b.operations {
                opnames.push(op.opname.clone());
                if op.opname == "getarrayitem" {
                    if let Hlvalue::Constant(c) = &op.args[0] {
                        hex_chars = Some(c.value.clone());
                    }
                }
            }
            for link in &b.exits {
                if let Some(target) = link.borrow().target.as_ref() {
                    queue.push(target.clone());
                }
            }
        }

        for expected in ["uint_and", "uint_rshift", "getarrayitem", "setarrayitem"] {
            assert!(
                opnames.iter().any(|n| n == expected),
                "ll_int2hex graph missing op {expected}; saw {opnames:?}"
            );
        }

        let Some(ConstValue::LLPtr(ptr)) = hex_chars else {
            panic!("digit lookup must read from the hex_chars table");
        };
        let Ok(Some(_ptr_obj::Array(arr))) = ptr._obj0_value() else {
            panic!("hex_chars LLPtr must target an Array container");
        };
        let items = arr.items.lock().unwrap();
        assert_eq!(items[0], LowLevelValue::Char('0'));
        assert_eq!(items[10], LowLevelValue::Char('a'));
        assert_eq!(items[15], LowLevelValue::Char('f'));
    }
}
