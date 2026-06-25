//! RPython `rpython/rtyper/lltypesystem/rbytearray.py`.
//!
//! This slice lands the low-level `BYTEARRAY` data shape and the
//! module-global `bytearray_repr`. Higher-level pairtype operations
//! (`bytearray + str`, indexed mutation) are kept in the top-level
//! `rbytearray` parity surface and can land independently.

use std::sync::{Arc, LazyLock, OnceLock};

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, _ptr_obj, Array, ForwardReference, LowLevelType, LowLevelValue, MallocFlavor, Ptr,
    PtrTarget, Struct, malloc, nullptr,
};
use crate::translator::rtyper::lltypesystem::rstr::{STRPTR, chars_array_ptr_lltype_from_strptr};
use crate::translator::rtyper::rmodel::{Repr, ReprState, gc_flavor_const, lowlevel_type_const};
use crate::translator::rtyper::rtyper::{
    constant_with_lltype, helper_pygraph_from_graph, variable_with_lltype,
};

/// RPython `BYTEARRAY = lltype.GcForwardReference()` resolved via
/// `BYTEARRAY.become(GcStruct('rpy_bytearray', ('chars', Array(Char)),
/// adtmeths={...}))`.
pub static BYTEARRAY: LazyLock<LowLevelType> = LazyLock::new(|| {
    let body = Struct::gc(
        "rpy_bytearray",
        vec![(
            "chars".to_string(),
            LowLevelType::Array(Box::new(Array::new(LowLevelType::Char))),
        )],
    );
    let fwd = ForwardReference::gc();
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("BYTEARRAY.become should succeed");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(BYTEARRAY)`.
pub static BYTEARRAYPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let LowLevelType::ForwardReference(fwd) = BYTEARRAY.clone() else {
        panic!("BYTEARRAY must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
});

fn bytearray_struct_lltype() -> Result<LowLevelType, TyperError> {
    let LowLevelType::ForwardReference(fwd) = BYTEARRAY.clone() else {
        return Err(TyperError::message("BYTEARRAY must be a ForwardReference"));
    };
    fwd.resolved()
        .ok_or_else(|| TyperError::message("BYTEARRAY forward reference unresolved"))
}

fn chars_array_ptr_lltype_from_bytearrayptr() -> Result<LowLevelType, TyperError> {
    let body = bytearray_struct_lltype()?;
    let LowLevelType::Struct(st) = body else {
        return Err(TyperError::message(
            "BYTEARRAY ForwardReference must resolve to Struct",
        ));
    };
    let chars_field = st
        ._flds
        .get("chars")
        .ok_or_else(|| TyperError::message("BYTEARRAY struct has no chars field"))?;
    let LowLevelType::Array(arr) = chars_field else {
        return Err(TyperError::message(
            "BYTEARRAY chars field must be Array(Char)",
        ));
    };
    Ok(LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array((**arr).clone()),
    })))
}

/// RPython `mallocbytearray(size): return lltype.malloc(BYTEARRAY, size)`.
pub fn mallocbytearray(size: usize) -> Result<_ptr, String> {
    let body = bytearray_struct_lltype().map_err(|e| e.to_string())?;
    malloc(body, Some(size), MallocFlavor::Gc, false)
}

/// RPython `empty = lltype.malloc(BYTEARRAY, 0, immortal=True)`.
pub fn empty() -> _ptr {
    static EMPTY: OnceLock<_ptr> = OnceLock::new();
    EMPTY
        .get_or_init(|| {
            let body = bytearray_struct_lltype().expect("BYTEARRAY resolved");
            malloc(body, Some(0), MallocFlavor::Gc, true).expect("empty BYTEARRAY malloc")
        })
        .clone()
}

/// RPython `_empty_bytearray(): return empty`.
pub fn _empty_bytearray() -> _ptr {
    empty()
}

/// RPython `class ByteArrayRepr(AbstractByteArrayRepr)`.
#[derive(Debug)]
pub struct ByteArrayRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl ByteArrayRepr {
    pub fn new() -> Self {
        ByteArrayRepr {
            state: ReprState::new(),
            lltype: BYTEARRAYPTR.clone(),
        }
    }

    /// RPython `ByteArrayRepr.ll_str(ll_b)`: copy bytearray chars into
    /// a freshly allocated `STR`, returning `nullptr(STR)` for null.
    pub fn ll_str(&self, ll_b: &_ptr) -> Result<_ptr, String> {
        if !ll_b.nonzero() {
            return Ok(crate::translator::rtyper::lltypesystem::rstr::null_str_ptr());
        }
        let bytes = hlbytearray(ll_b)?;
        crate::translator::rtyper::lltypesystem::rstr::llstr(&bytes)
    }
}

impl Default for ByteArrayRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for ByteArrayRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ByteArrayRepr"
    }

    fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
        crate::translator::rtyper::pairtype::ReprClassId::ByteArrayRepr
    }

    /// RPython `ByteArrayRepr.convert_const`:
    /// `None -> nullptr(BYTEARRAY)`, otherwise allocate and copy each
    /// byte into `chars`.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        match value {
            ConstValue::None => Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(
                    nullptr(BYTEARRAY.clone()).expect("nullptr(BYTEARRAY) must succeed"),
                )),
                self.lltype.clone(),
            )),
            ConstValue::ByteStr(bytes) => {
                let p = mallocbytearray(bytes.len()).map_err(TyperError::message)?;
                fill_bytearray_chars(&p, bytes).map_err(TyperError::message)?;
                Ok(Constant::with_concretetype(
                    ConstValue::LLPtr(Box::new(p)),
                    self.lltype.clone(),
                ))
            }
            other => Err(TyperError::message(format!("not a bytearray: {other:?}"))),
        }
    }
}

/// RPython `bytearray_repr = ByteArrayRepr()`.
pub fn bytearray_repr() -> Arc<ByteArrayRepr> {
    static REPR: OnceLock<Arc<ByteArrayRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(ByteArrayRepr::new())).clone()
}

fn fill_bytearray_chars(p: &_ptr, bytes: &[u8]) -> Result<(), String> {
    let _ptr_obj::Struct(st) = p
        ._obj0_value()
        .map_err(|_| "fill_bytearray_chars: delayed pointer".to_string())?
        .ok_or_else(|| "fill_bytearray_chars: null pointer".to_string())?
    else {
        return Err("fill_bytearray_chars: expected Struct container".to_string());
    };
    let fields = st._fields.lock().unwrap();
    let Some((_, LowLevelValue::Array(chars))) = fields.iter().find(|(n, _)| n == "chars") else {
        return Err("fill_bytearray_chars: rpy_bytearray lacks chars".to_string());
    };
    for (i, b) in bytes.iter().enumerate() {
        if !chars.setitem(i, LowLevelValue::Char(*b as char)) {
            return Err(format!(
                "fill_bytearray_chars: chars[{i}] write out of bounds"
            ));
        }
    }
    Ok(())
}

/// RPython `hlbytearray(ll_b)`.
pub fn hlbytearray(ll_b: &_ptr) -> Result<Vec<u8>, String> {
    if !ll_b.nonzero() {
        return Err("hlbytearray: null bytearray pointer".to_string());
    }
    let _ptr_obj::Struct(st) = ll_b
        ._obj0_value()
        .map_err(|_| "hlbytearray: delayed pointer".to_string())?
        .ok_or_else(|| "hlbytearray: null bytearray pointer".to_string())?
    else {
        return Err("hlbytearray: expected Struct container".to_string());
    };
    let fields = st._fields.lock().unwrap();
    let Some((_, LowLevelValue::Array(chars))) = fields.iter().find(|(n, _)| n == "chars") else {
        return Err("hlbytearray: rpy_bytearray lacks chars".to_string());
    };
    let mut bytes = Vec::with_capacity(chars.getlength());
    for i in 0..chars.getlength() {
        match chars.getitem(i) {
            Some(LowLevelValue::Char(c)) => bytes.push(c as u32 as u8),
            other => {
                return Err(format!(
                    "hlbytearray: chars[{i}] expected Char, got {other:?}"
                ));
            }
        }
    }
    Ok(bytes)
}

fn bool_const(value: bool) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::Bool(value),
        LowLevelType::Bool,
    ))
}

fn signed_const(n: i64) -> Hlvalue {
    constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed)
}

/// Synthesize `LLHelpers.ll_str2bytearray` from
/// `lltypesystem/rstr.py:385-391`.
pub(crate) fn build_ll_str2bytearray_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    let src_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let dst_chars_ptr_lltype = chars_array_ptr_lltype_from_bytearrayptr()?;
    let bytearray_struct = bytearray_struct_lltype()?;
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("str", STRPTR.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", BYTEARRAYPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let src_chars_for_cond = variable_with_lltype("src_chars", src_chars_ptr_lltype.clone());
    let result_for_cond = variable_with_lltype("result", BYTEARRAYPTR.clone());
    let dst_chars_for_cond = variable_with_lltype("dst_chars", dst_chars_ptr_lltype.clone());
    let len_for_cond = variable_with_lltype("lgt", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_cond.clone()),
        Hlvalue::Variable(result_for_cond.clone()),
        Hlvalue::Variable(dst_chars_for_cond.clone()),
        Hlvalue::Variable(len_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let src_chars_for_body = variable_with_lltype("src_chars", src_chars_ptr_lltype.clone());
    let result_for_body = variable_with_lltype("result", BYTEARRAYPTR.clone());
    let dst_chars_for_body = variable_with_lltype("dst_chars", dst_chars_ptr_lltype.clone());
    let len_for_body = variable_with_lltype("lgt", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_body.clone()),
        Hlvalue::Variable(result_for_body.clone()),
        Hlvalue::Variable(dst_chars_for_body.clone()),
        Hlvalue::Variable(len_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    let src_chars = variable_with_lltype("src_chars", src_chars_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(src_chars.clone()),
    ));
    let lgt = variable_with_lltype("lgt", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(src_chars.clone())],
        Hlvalue::Variable(lgt.clone()),
    ));
    let result = variable_with_lltype("result", BYTEARRAYPTR.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(bytearray_struct),
            gc_flavor_const()?,
            Hlvalue::Variable(lgt.clone()),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    let dst_chars = variable_with_lltype("dst_chars", dst_chars_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(result.clone()), chars_field_const()],
        Hlvalue::Variable(dst_chars.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars),
                Hlvalue::Variable(result),
                Hlvalue::Variable(dst_chars),
                Hlvalue::Variable(lgt),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(len_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_cond),
                Hlvalue::Variable(result_for_cond.clone()),
                Hlvalue::Variable(dst_chars_for_cond),
                Hlvalue::Variable(len_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(result_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let ch = variable_with_lltype("ch", LowLevelType::Char);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(src_chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(dst_chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
                Hlvalue::Variable(ch),
            ],
            Hlvalue::Variable(set_void),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_body),
                Hlvalue::Variable(result_for_body),
                Hlvalue::Variable(dst_chars_for_body),
                Hlvalue::Variable(len_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["str".to_string()],
        func,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn _empty_bytearray_returns_module_empty_singleton() {
        let direct = empty();
        let through_adtmeth = _empty_bytearray();
        assert_eq!(hlbytearray(&through_adtmeth).unwrap(), Vec::<u8>::new());
        assert_eq!(
            direct._hashable_identity(),
            through_adtmeth._hashable_identity()
        );
    }

    #[test]
    fn bytearray_repr_convert_const_allocates_chars() {
        let repr = ByteArrayRepr::new();
        let c = repr
            .convert_const(&ConstValue::ByteStr(b"abc".to_vec()))
            .expect("convert_const bytearray");
        let ConstValue::LLPtr(ptr) = c.value else {
            panic!("bytearray constant must be an LLPtr");
        };
        assert_eq!(hlbytearray(&ptr).unwrap(), b"abc");
        assert_eq!(c.concretetype, Some(BYTEARRAYPTR.clone()));
    }

    #[test]
    fn bytearray_repr_convert_const_none_is_null_ptr() {
        let repr = ByteArrayRepr::new();
        let c = repr.convert_const(&ConstValue::None).expect("None");
        let ConstValue::LLPtr(ptr) = c.value else {
            panic!("None bytearray must be an LLPtr");
        };
        assert!(!ptr.nonzero());
        assert_eq!(c.concretetype, Some(BYTEARRAYPTR.clone()));
    }

    #[test]
    fn build_ll_str2bytearray_graph_matches_source_loop_shape() {
        let helper = build_ll_str2bytearray_helper_graph("ll_str2bytearray").unwrap();
        let inner = helper.graph.borrow();
        let sb = inner.startblock.borrow();
        let start_ops: Vec<_> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "malloc_varsize",
                "getsubstruct"
            ]
        );
        let loop_cond = sb.exits[0].borrow().target.as_ref().unwrap().clone();
        assert_eq!(loop_cond.borrow().operations[0].opname, "int_lt");
        let loop_body = loop_cond.borrow().exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        let body_ops: Vec<_> = loop_body
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(body_ops, vec!["getarrayitem", "setarrayitem", "int_add"]);
    }
}
