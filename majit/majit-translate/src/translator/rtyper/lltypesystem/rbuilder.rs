//! RPython `rpython/rtyper/lltypesystem/rbuilder.py` parity module.
//!
//! This slice lands the low-level builder container shapes and repr
//! class names. The append/grow/build helper graphs are still pending,
//! but the public lltype names now match upstream:
//! `STRINGPIECE`, `STRINGBUILDER`, `UNICODEPIECE`, `UNICODEBUILDER`,
//! `BaseStringBuilderRepr`, `StringBuilderRepr`, and
//! `UnicodeBuilderRepr`.

#![allow(non_snake_case, non_upper_case_globals)]

use std::sync::LazyLock;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    ForwardReference, LowLevelType, Ptr, PtrTarget, Struct,
};
use crate::translator::rtyper::lltypesystem::rstr::{STRPTR, UNICODEPTR};
use crate::translator::rtyper::rtyper::{
    constant_with_lltype, helper_pygraph_from_graph, variable_with_lltype, void_field_const,
};

fn ptr_to_lowlevel(target: LowLevelType) -> LowLevelType {
    match target {
        LowLevelType::ForwardReference(fwd) => LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(*fwd),
        })),
        LowLevelType::Struct(t) => LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(*t),
        })),
        other => panic!("expected container lowleveltype for Ptr(...), got {other:?}"),
    }
}

fn ptr_to_forward_reference(target: &LowLevelType) -> LowLevelType {
    let LowLevelType::ForwardReference(fwd) = target.clone() else {
        panic!("builder piece type must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
}

/// RPython `STRINGPIECE = lltype.GcStruct('stringpiece', ...)`.
pub static STRINGPIECE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let fwd = ForwardReference::gc();
    let body = Struct::gc(
        "stringpiece",
        vec![
            ("buf".into(), STRPTR.clone()),
            (
                "prev_piece".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd.clone()),
                })),
            ),
        ],
    );
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("STRINGPIECE.prev_piece.TO.become(STRINGPIECE)");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(STRINGPIECE)`.
pub static STRINGPIECEPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_forward_reference(&STRINGPIECE));

/// RPython `STRINGBUILDER = lltype.GcStruct('stringbuilder', ...)`.
pub static STRINGBUILDER: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(Struct::gc(
        "stringbuilder",
        vec![
            ("current_buf".into(), STRPTR.clone()),
            ("current_pos".into(), LowLevelType::Signed),
            ("current_end".into(), LowLevelType::Signed),
            ("total_size".into(), LowLevelType::Signed),
            ("extra_pieces".into(), STRINGPIECEPTR.clone()),
        ],
    )))
});

/// RPython `Ptr(STRINGBUILDER)`.
pub static STRINGBUILDERPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_lowlevel(STRINGBUILDER.clone()));

/// RPython `UNICODEPIECE = lltype.GcStruct('unicodepiece', ...)`.
pub static UNICODEPIECE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let fwd = ForwardReference::gc();
    let body = Struct::gc(
        "unicodepiece",
        vec![
            ("buf".into(), UNICODEPTR.clone()),
            (
                "prev_piece".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd.clone()),
                })),
            ),
        ],
    );
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("UNICODEPIECE.prev_piece.TO.become(UNICODEPIECE)");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(UNICODEPIECE)`.
pub static UNICODEPIECEPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_forward_reference(&UNICODEPIECE));

/// RPython `UNICODEBUILDER = lltype.GcStruct('unicodebuilder', ...)`.
pub static UNICODEBUILDER: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(Struct::gc(
        "unicodebuilder",
        vec![
            ("current_buf".into(), UNICODEPTR.clone()),
            ("current_pos".into(), LowLevelType::Signed),
            ("current_end".into(), LowLevelType::Signed),
            ("total_size".into(), LowLevelType::Signed),
            ("extra_pieces".into(), UNICODEPIECEPTR.clone()),
        ],
    )))
});

/// RPython `Ptr(UNICODEBUILDER)`.
pub static UNICODEBUILDERPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_lowlevel(UNICODEBUILDER.clone()));

fn builder_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rbuilder.{name} - low-level StringBuilder runtime helper deferred"
    ))
}

pub fn _ll_append() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("_ll_append"))
}

pub fn ll_grow_by() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_grow_by"))
}

pub fn ll_grow_and_append() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_grow_and_append"))
}

pub fn ll_append() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append"))
}

pub fn ll_jit_append() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_jit_append"))
}

pub fn ll_append_res0() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_res0"))
}

pub fn ll_append_char() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_char"))
}

pub fn ll_append_slice() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_slice"))
}

pub fn ll_jit_append_slice() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_jit_append_slice"))
}

pub fn ll_append_res_slice() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_res_slice"))
}

pub const MAX_N: usize = 10;

pub fn make_func_for_size(N: usize) -> (String, String, usize) {
    (
        format!("ll_append_0_{N}"),
        format!("ll_append_start_{N}"),
        N,
    )
}

pub static unroll_func_for_size: LazyLock<Vec<(String, String, usize)>> =
    LazyLock::new(|| (2..=MAX_N).map(make_func_for_size).collect());

pub fn ll_jit_try_append_slice() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_jit_try_append_slice"))
}

pub fn ll_append_multiple_char() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_multiple_char"))
}

pub fn _ll_append_multiple_char() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("_ll_append_multiple_char"))
}

pub fn ll_jit_try_append_multiple_char() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_jit_try_append_multiple_char"))
}

pub fn ll_append_charpsize() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_charpsize"))
}

pub fn ll_getlength() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_getlength"))
}

/// Synthesise `ll_getlength(ll_builder)` (`rbuilder.py:347-350`):
/// `ll_builder.total_size - (ll_builder.current_end - ll_builder.current_pos)`.
pub fn build_ll_getlength_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(ll_builder.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let current_end = variable_with_lltype("current_end", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(current_end.clone()),
    ));
    let current_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(current_pos.clone()),
    ));
    let num_chars_missing_from_last_piece =
        variable_with_lltype("num_chars_missing_from_last_piece", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(current_end),
            Hlvalue::Variable(current_pos),
        ],
        Hlvalue::Variable(num_chars_missing_from_last_piece.clone()),
    ));
    let total_size = variable_with_lltype("total_size", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder),
            void_field_const("total_size"),
        ],
        Hlvalue::Variable(total_size.clone()),
    ));
    let result = variable_with_lltype("result", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(total_size),
            Hlvalue::Variable(num_chars_missing_from_last_piece),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
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
        vec!["ll_builder".to_string()],
        func,
    ))
}

pub fn ll_build() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_build"))
}

pub fn ll_shrink_final() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_shrink_final"))
}

pub fn ll_fold_pieces() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_fold_pieces"))
}

pub fn ll_bool() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_bool"))
}

/// Synthesise `ll_bool(ll_builder)` (`rbuilder.py:417-418`):
/// `ll_builder != nullptr(lltype.typeOf(ll_builder).TO)`.
pub fn build_ll_bool_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(ll_builder.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // result = ptr_ne(ll_builder, nullptr(TO))
    let null_builder = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        builder_ptr_lltype,
    ));
    let result = variable_with_lltype("result", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_ne",
        vec![Hlvalue::Variable(ll_builder), null_builder],
        Hlvalue::Variable(result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
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
        vec!["ll_builder".to_string()],
        func,
    ))
}

/// Synthesise `ll_new(init_size)` (`rbuilder.py:446-455` / `469-478`):
///
/// ```python
/// init_size = intmask(min(r_uint(init_size), r_uint(1280)))
/// ll_builder = lltype.malloc(STRINGBUILDER)
/// ll_builder.current_buf = ll_builder.mallocfn(init_size)
/// ll_builder.current_pos = 0
/// ll_builder.current_end = init_size
/// ll_builder.total_size = init_size
/// return ll_builder
/// ```
///
/// `min` is `rbuiltin.ll_min` (`rbuiltin.py:238`) and `mallocfn` is the
/// specialization's `staticAdtMethod(rstr.mallocstr / mallocunicode)`
/// (`rbuilder.py:54`/`72`) — both baked in as `direct_call` callee consts,
/// mirroring [`build_ll_call_lookup_function_helper_graph`]. `buf_lltype`
/// is `STRPTR`/`UNICODEPTR` (the `current_buf` field and `mallocfn` result).
pub fn build_ll_new_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    builder_struct: LowLevelType,
    buf_lltype: LowLevelType,
    min_fn: Constant,
    mallocfn: Constant,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let init_size = variable_with_lltype("init_size", LowLevelType::Signed);
    let startblock = Block::shared(vec![Hlvalue::Variable(init_size.clone())]);
    let return_var = variable_with_lltype("result", builder_ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let void_result = || variable_with_lltype("v", LowLevelType::Void);

    // init_size = intmask(min(r_uint(init_size), r_uint(1280)))
    let uint_size = variable_with_lltype("uint_size", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(init_size)],
        Hlvalue::Variable(uint_size.clone()),
    ));
    let uint_min = variable_with_lltype("uint_min", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(min_fn),
            Hlvalue::Variable(uint_size),
            constant_with_lltype(ConstValue::Int(1280), LowLevelType::Unsigned),
        ],
        Hlvalue::Variable(uint_min.clone()),
    ));
    let size = variable_with_lltype("size", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_uint_to_int",
        vec![Hlvalue::Variable(uint_min)],
        Hlvalue::Variable(size.clone()),
    ));

    // ll_builder = lltype.malloc(STRINGBUILDER)
    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![lowlevel_type_const(builder_struct), gc_flavor_const()?],
        Hlvalue::Variable(ll_builder.clone()),
    ));

    // ll_builder.current_buf = ll_builder.mallocfn(init_size)
    let current_buf = variable_with_lltype("current_buf", buf_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(mallocfn), Hlvalue::Variable(size.clone())],
        Hlvalue::Variable(current_buf.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_buf"),
            Hlvalue::Variable(current_buf),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // ll_builder.current_pos = 0
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // ll_builder.current_end = init_size
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
            Hlvalue::Variable(size.clone()),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // ll_builder.total_size = init_size
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("total_size"),
            Hlvalue::Variable(size),
        ],
        Hlvalue::Variable(void_result()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(ll_builder)],
            Some(graph.returnblock.clone()),
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
        vec!["init_size".to_string()],
        func,
    ))
}

/// RPython `class BaseStringBuilderRepr(AbstractStringBuilderRepr)`.
#[derive(Debug, Default)]
pub struct BaseStringBuilderRepr;

/// RPython `class StringBuilderRepr(BaseStringBuilderRepr)`.
#[derive(Debug, Default)]
pub struct StringBuilderRepr;

impl StringBuilderRepr {
    /// RPython `StringBuilderRepr.lowleveltype = lltype.Ptr(STRINGBUILDER)`.
    pub fn lowleveltype(&self) -> &'static LowLevelType {
        &STRINGBUILDERPTR
    }

    /// RPython `StringBuilderRepr.basetp = STR`.
    pub fn basetp(&self) -> &'static LowLevelType {
        &crate::translator::rtyper::lltypesystem::rstr::STR
    }
}

/// RPython `class UnicodeBuilderRepr(BaseStringBuilderRepr)`.
#[derive(Debug, Default)]
pub struct UnicodeBuilderRepr;

impl UnicodeBuilderRepr {
    /// RPython `UnicodeBuilderRepr.lowleveltype = lltype.Ptr(UNICODEBUILDER)`.
    pub fn lowleveltype(&self) -> &'static LowLevelType {
        &UNICODEBUILDERPTR
    }

    /// RPython `UnicodeBuilderRepr.basetp = UNICODE`.
    pub fn basetp(&self) -> &'static LowLevelType {
        &crate::translator::rtyper::lltypesystem::rstr::UNICODE
    }
}

static STRINGBUILDER_REPR: LazyLock<StringBuilderRepr> = LazyLock::new(StringBuilderRepr::default);
static UNICODEBUILDER_REPR: LazyLock<UnicodeBuilderRepr> =
    LazyLock::new(UnicodeBuilderRepr::default);

/// RPython `stringbuilder_repr = StringBuilderRepr()`.
pub fn stringbuilder_repr() -> &'static StringBuilderRepr {
    &STRINGBUILDER_REPR
}

/// RPython `unicodebuilder_repr = UnicodeBuilderRepr()`.
pub fn unicodebuilder_repr() -> &'static UnicodeBuilderRepr {
    &UNICODEBUILDER_REPR
}

#[cfg(test)]
mod tests {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};

    #[test]
    fn stringpiece_prev_piece_points_back_to_stringpiece() {
        let LowLevelType::ForwardReference(piece_fwd) = super::STRINGPIECE.clone() else {
            panic!("STRINGPIECE must be a GcForwardReference");
        };
        let Some(LowLevelType::Struct(piece)) = piece_fwd.resolved() else {
            panic!("STRINGPIECE must resolve to a struct");
        };
        let Some(LowLevelType::Ptr(prev_ptr)) = piece.getattr_field_type("prev_piece") else {
            panic!("STRINGPIECE.prev_piece must be Ptr");
        };
        assert!(matches!(prev_ptr.TO, PtrTarget::ForwardReference(_)));
        assert_eq!(
            LowLevelType::ForwardReference(piece_fwd),
            LowLevelType::from(prev_ptr.TO)
        );
    }

    #[test]
    fn stringbuilder_fields_match_rpython_shape() {
        let LowLevelType::Struct(builder) = super::STRINGBUILDER.clone() else {
            panic!("STRINGBUILDER must be GcStruct");
        };
        assert_eq!(
            builder._names,
            vec![
                "current_buf",
                "current_pos",
                "current_end",
                "total_size",
                "extra_pieces",
            ]
        );
        assert_eq!(
            builder.getattr_field_type("current_pos"),
            Some(LowLevelType::Signed)
        );
        assert_eq!(
            builder.getattr_field_type("current_end"),
            Some(LowLevelType::Signed)
        );
        assert_eq!(
            builder.getattr_field_type("total_size"),
            Some(LowLevelType::Signed)
        );
    }

    #[test]
    fn repr_singletons_have_distinct_lowleveltypes() {
        assert_ne!(
            super::stringbuilder_repr().lowleveltype(),
            super::unicodebuilder_repr().lowleveltype()
        );
    }

    #[test]
    fn runtime_helper_surface_is_explicitly_deferred() {
        let err = super::ll_append().expect_err("append helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_append"));

        let err = super::ll_build().expect_err("build helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_build"));

        let err = super::ll_append_multiple_char().expect_err("multiple-char helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_append_multiple_char"));
    }

    #[test]
    fn build_ll_getlength_reads_fields_and_returns_signed_length() {
        use super::Hlvalue;
        let helper =
            super::build_ll_getlength_helper_graph("ll_getlength", super::STRINGBUILDERPTR.clone())
                .expect("build_ll_getlength_helper_graph");
        assert_eq!(helper.func.name, "ll_getlength");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        // total_size - (current_end - current_pos)
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec!["getfield", "getfield", "int_sub", "getfield", "int_sub"]
        );
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed)
        );
    }

    #[test]
    fn build_ll_bool_compares_pointer_against_null_and_returns_bool() {
        use super::Hlvalue;
        let helper = super::build_ll_bool_helper_graph("ll_bool", super::STRINGBUILDERPTR.clone())
            .expect("build_ll_bool_helper_graph");
        assert_eq!(helper.func.name, "ll_bool");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        // ll_builder != nullptr(TO)
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["ptr_ne"]);
        assert_eq!(startblock.inputargs.len(), 1);
        // second arg is the null pointer constant of the builder's own type.
        let Hlvalue::Constant(null_arg) = &startblock.operations[0].args[1] else {
            panic!("ptr_ne second arg must be a null Constant");
        };
        assert_eq!(null_arg.value, super::ConstValue::None);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Bool));
    }

    fn dummy_funcptr_const() -> super::Constant {
        super::Constant::with_concretetype(super::ConstValue::None, LowLevelType::Void)
    }

    #[test]
    fn build_ll_new_clamps_size_mallocs_builder_and_inits_fields() {
        use super::Hlvalue;
        let helper = super::build_ll_new_helper_graph(
            "ll_new",
            super::STRINGBUILDERPTR.clone(),
            super::STRINGBUILDER.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_new_helper_graph");
        assert_eq!(helper.func.name, "ll_new");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        // intmask(min(r_uint(init_size), 1280)); malloc; mallocfn; 4 setfields
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec![
                "cast_int_to_uint",
                "direct_call",
                "cast_uint_to_int",
                "malloc",
                "direct_call",
                "setfield",
                "setfield",
                "setfield",
                "setfield",
            ]
        );
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(super::STRINGBUILDERPTR.clone())
        );
    }

    #[test]
    fn jit_specialized_size_table_matches_upstream_range() {
        assert_eq!(super::MAX_N, 10);
        assert_eq!(
            super::make_func_for_size(2),
            ("ll_append_0_2".into(), "ll_append_start_2".into(), 2)
        );
        assert_eq!(super::unroll_func_for_size.len(), 9);
        assert_eq!(super::unroll_func_for_size[0].2, 2);
        assert_eq!(super::unroll_func_for_size.last().unwrap().2, super::MAX_N);
    }
}
