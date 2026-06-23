//! RPython `rpython/rtyper/lltypesystem/ll_str.py` parity module.
//!
//! Pyre currently materialises the ported `ll_int2hex` body as a helper
//! graph builder in [`super::rstr`], because callers need a `direct_call`
//! target rather than a host-side Rust string conversion.  Keep the
//! upstream file path available here while re-exporting only the helper
//! surface that is actually implemented.

pub use crate::translator::rtyper::lltypesystem::rstr::build_ll_int2hex_helper_graph;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Hlvalue};
    use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, LowLevelValue};

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
