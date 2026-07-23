//! Translation coverage for the RPython `rbigint` port.
//!
//! `RBigInt` is part of the interpreter's translated program, not a foreign
//! arithmetic library.  Its inherent methods therefore have to remain real
//! flow graphs under the same `[module::RBigInt, method]` keys emitted at
//! call sites.

use majit_charon_reader::Llbc;
use majit_translate::{
    HostStaticAddrs,
    front::{
        llbc_hints::harvest_hints_from_llbcs,
        mir::build_semantic_program_from_llbcs_with_static_addrs_and_module_paths,
    },
    model::{CallTarget, OpKind, ValueType},
};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);
const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);
const RBIGINT_RS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../pyre/pyre-object/src/rbigint.rs"
);

fn assert_source_order(source: &str, fragments: &[&str]) {
    let mut cursor = 0;
    for fragment in fragments {
        let relative = source[cursor..]
            .find(fragment)
            .unwrap_or_else(|| panic!("missing `{fragment}` after byte {cursor}"));
        cursor += relative + fragment.len();
    }
}

#[test]
fn mapped_rbigint_methods_and_helpers_follow_upstream_source_order() {
    let source = std::fs::read_to_string(RBIGINT_RS).expect("read Rust rbigint port");
    let core_start = source.find("impl RBigInt {").expect("primary RBigInt impl");
    let compatibility_start = source
        .find("/// Rust ecosystem compatibility surface")
        .expect("compatibility RBigInt impl");
    let core = &source[core_start..compatibility_start];

    // rpython/rlib/rbigint.py:99-114.  Each suffix is one concrete graph
    // emitted by the corresponding `@specialize.argtype(0)` decorator.
    assert_source_order(
        &source[..core_start],
        &[
            "fn _mask_digit(",
            "fn _mask_udigit(",
            "fn _mask_widedigit(",
            "fn _mask_uwidedigit(",
            "fn _widen_digit(",
            "fn _unsigned_widen_digit(",
            "fn _store_digit(",
            "fn _store_udigit(",
            "fn _store_widedigit(",
            "fn _store_uwidedigit(",
            "fn _load_unsigned_digit(",
        ],
    );

    // rpython/rlib/rbigint.py:168-1631. Rust-only allocation and storage
    // helpers may appear between these methods, but every translated upstream
    // method must retain the original relative order. Python special methods
    // represented by Rust traits are deliberately outside this inherent impl.
    assert_source_order(
        core,
        &[
            "fn get_sign(",
            "fn _set_sign(",
            "fn digit(",
            "fn widedigit(",
            "fn uwidedigit(",
            "fn udigit(",
            "fn setdigit(",
            "fn setdigit_udigit(",
            "fn setdigit_widedigit(",
            "fn setdigit_uwidedigit(",
            "fn numdigits(",
            "fn fromint(",
            "fn frombool(",
            "fn fromlong(",
            "fn fromfloat(",
            "fn _fromfloat_finite(",
            "fn fromrarith_int(",
            "fn fromrarith_uint(",
            "fn fromdecimalstr(",
            "fn fromstr(",
            "fn _from_numberstring_parser(",
            "fn frombytes(",
            "fn tobytes(",
            "fn toint(",
            "fn _toint_helper(",
            "fn fits_int(",
            "fn tolonglong(",
            "fn tobool(",
            "fn touint(",
            "fn _touint_helper(",
            "fn toulonglong(",
            "fn uintmask(",
            "fn ulonglongmask(",
            "fn tofloat(",
            "fn format(",
            "fn repr(",
            "fn str(",
            "fn eq(",
            "fn int_eq(",
            "fn ne(",
            "fn int_ne(",
            "fn lt(",
            "fn int_lt(",
            "fn le(",
            "fn int_le(",
            "fn gt(",
            "fn int_gt(",
            "fn ge(",
            "fn int_ge(",
            "fn hash(",
            "fn add(",
            "fn int_add(",
            "fn add_int_int_bigint_result(",
            "fn _add_int_int_helper(",
            "fn sub(",
            "fn int_sub(",
            "fn sub_int_int_bigint_result(",
            "fn mul(",
            "fn int_mul(",
            "fn mul_int_int_bigint_result(",
            "fn truediv(",
            "fn floordiv(",
            "fn div(",
            "fn int_floordiv(",
            "fn int_div(",
            "fn r#mod(",
            "fn int_mod(",
            "fn int_mod_int_result(",
            "fn divmod(",
            "fn _divmod_small(",
            "fn int_divmod(",
            "fn pow(",
            "fn int_pow(",
            "fn neg(",
            "fn abs(",
            "fn invert(",
            "fn lshift(",
            "fn lqshift(",
            "fn lshift_int_int_bigint_result(",
            "fn rshift(",
            "fn rqshift(",
            "fn abs_rshift_and_mask(",
            "fn from_list_n_bits(",
            "fn and_(",
            "fn int_and_(",
            "fn xor(",
            "fn int_xor(",
            "fn or_(",
            "fn int_or_(",
            "fn oct(",
            "fn hex(",
            "fn log(",
            "fn tolong(",
            "fn _normalize(",
            "fn bit_length(",
            "fn bit_count(",
            "fn gcd(",
            "fn isqrt(",
        ],
    );
    assert!(
        !core.contains("fn from_u128("),
        "Rust ecosystem compatibility API leaked into the upstream method block"
    );
    for signed_result in [
        "pub fn numdigits(&self) -> i64",
        "pub fn bit_length(&self) -> Result<i64, RBigIntError>",
        "pub fn bit_count(&self) -> Result<i64, RBigIntError>",
    ] {
        assert!(
            core.contains(signed_result),
            "upstream Signed result changed shape: {signed_result}"
        );
    }

    // rpython/rlib/rbigint.py:1671-3814. Owner structs and Rust GC hooks are
    // allowed between mapped helpers; the upstream algorithm sequence is not.
    let helpers = &source[compatibility_start..];
    assert_source_order(
        helpers,
        &[
            "fn _help_mult(",
            "fn digits_from_nonneg_long(",
            "fn digits_from_nonneg_ulong(",
            "fn digits_for_most_neg_long(",
            "fn args_from_rarith_int1(",
            "fn args_from_rarith_int(",
            "fn args_from_rarith_uint1(",
            "fn args_from_rarith_uint(",
            "fn args_from_long(",
            "fn _x_add",
            "fn _x_int_add(",
            "fn _x_sub",
            "fn _x_int_sub(",
            "fn _x_mul(",
            "fn _kmul_split(",
            "fn _k_mul(",
            "fn _inplace_divrem1(",
            "fn _divrem1(",
            "fn _int_rem_core(",
            "fn _v_iadd(",
            "fn _v_isub(",
            "fn _muladd1(",
            "fn _v_lshift(",
            "fn _v_rshift(",
            "fn _x_divrem(",
            "fn _divrem(",
            "fn _extract_digits(",
            "fn div2n1n(",
            "fn div3n2n(",
            "fn _full_digits_lshift_then_or(",
            "fn _divmod_fast_pos(",
            "fn divmod_big(",
            "fn _x_int_lt(",
            "fn _AsScaledDouble(",
            "fn _AsDouble(",
            "fn _loghelper_ln(",
            "fn _loghelper_log10(",
            "fn _loghelper_log2(",
            "fn bits_in_digit(",
            "fn bit_length_int(",
            "fn bit_count_digit(",
            "fn _bitcount64(",
            "fn _bitcount64_ops(",
            "fn _truediv_result(",
            "fn _truediv_overflow",
            "fn _bigint_true_divide(",
            "fn _format_base2_notzero(",
            "fn get_cached_parts(",
            "fn _format_int_general(",
            "fn _format_int10(",
            "fn _format_int10_18digits(",
            "fn _format_recursive_decimal(",
            "fn _format_recursive_general(",
            "fn _format_lowest_level_divmod_int_results(",
            "fn _format(",
            "fn _bitwise_and(",
            "fn _bitwise_or(",
            "fn _bitwise_xor(",
            "fn _int_bitwise_and(",
            "fn _int_bitwise_or(",
            "fn _int_bitwise_xor(",
            "fn _AsLongLong(",
            "fn _AsULonglong_ignore_sign(",
            "fn make_unsigned_mask_conversion(",
            "fn _As_unsigned_mask(",
            "fn _hash(",
            "fn digits_max_for_base(",
            "fn _decimalstr_to_bigint(",
            "fn parse_digit_string(",
            "fn _str_to_int_big_w5pow(",
            "fn _str_to_int_big_inner10(",
            "fn _str_to_int_big_base10(",
            "fn parse_string_from_binary_base(",
            "fn gcd_binary(",
            "fn lehmer_xgcd(",
            "fn gcd_lehmer(",
            "fn frombytes_int(",
            "fn tobytes_int(",
        ],
    );
    for runtime_discriminator in ["enum LogKind", "enum FormatIntKind", "enum BitOp"] {
        assert!(
            !helpers.contains(runtime_discriminator),
            "upstream @specialize.arg graph was collapsed behind {runtime_discriminator}"
        );
    }
}

#[test]
fn rbigint_impl_methods_preserve_upstream_elidable_markers() {
    if !std::path::Path::new(OBJECT_LLBC).is_file() {
        eprintln!(
            "skipping: {OBJECT_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-object`"
        );
        return;
    }

    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let hints = harvest_hints_from_llbcs(&[llbc]);
    // Exhaustive mapping of every `@jit.elidable` method in the pinned
    // rpython/rlib/rbigint.py.  This is a performance contract, not just
    // metadata: missing one changes whether callers residualize a pure bigint
    // operation or trace through its implementation. `fromrarith_uint` is the
    // unsigned Rust specialization of upstream's
    // `@specialize.argtype(0) fromrarith_int`.
    for name in [
        "fromint",
        "frombool",
        "fromfloat",
        "_fromfloat_finite",
        "fromrarith_int",
        "fromrarith_uint",
        "fromdecimalstr",
        "fromstr",
        "frombytes",
        "tobytes",
        "toint",
        "_toint_helper",
        "tolonglong",
        "touint",
        "_touint_helper",
        "toulonglong",
        "uintmask",
        "ulonglongmask",
        "tofloat",
        "format",
        "repr",
        "str",
        "eq",
        "int_eq",
        "lt",
        "int_lt",
        "hash",
        "add",
        "int_add",
        "add_int_int_bigint_result",
        "sub",
        "int_sub",
        "sub_int_int_bigint_result",
        "mul",
        "int_mul",
        "mul_int_int_bigint_result",
        "truediv",
        "int_floordiv",
        "int_mod",
        "int_mod_int_result",
        "divmod",
        "int_divmod",
        "pow",
        "int_pow",
        "neg",
        "abs",
        "invert",
        "lshift",
        "lqshift",
        "lshift_int_int_bigint_result",
        "rshift",
        "rqshift",
        "abs_rshift_and_mask",
        "and_",
        "int_and_",
        "xor",
        "int_xor",
        "or_",
        "int_or_",
        "oct",
        "hex",
        "log",
        "bit_length",
        "bit_count",
    ] {
        let path = format!("rbigint::<Impl>::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "elidable")),
            "missing upstream @jit.elidable marker for {path}: {hints:?}"
        );
    }
    // Exhaustive set of mapped core methods that upstream deliberately does
    // not mark elidable. They are projections, relational compositions,
    // mutators, or entry points whose pure callees carry the effect.
    for name in [
        "get_sign",
        "_set_sign",
        "digit",
        "widedigit",
        "uwidedigit",
        "udigit",
        "setdigit",
        "setdigit_udigit",
        "setdigit_widedigit",
        "setdigit_uwidedigit",
        "numdigits",
        "_from_numberstring_parser",
        "fits_int",
        "tobool",
        "ne",
        "int_ne",
        "le",
        "int_le",
        "gt",
        "int_gt",
        "ge",
        "int_ge",
        "_add_int_int_helper",
        "floordiv",
        "div",
        "mod",
        "_divmod_small",
        "int_div",
        "from_list_n_bits",
        "_normalize",
        "gcd",
        "isqrt",
    ] {
        let path = format!("rbigint::<Impl>::{name}");
        assert!(
            !hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "elidable")),
            "{path} is an upstream thin projection, not an @jit.elidable body"
        );
    }
    for name in [
        "_divrem1",
        "_divrem",
        "bit_length_int",
        "_bitcount64",
        "gcd_binary",
        "gcd_lehmer",
        "frombytes_int",
        "tobytes_int",
    ] {
        let path = format!("rbigint::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "elidable")),
            "missing upstream @jit.elidable marker for {path}: {hints:?}"
        );
    }
    for name in ["fromlong", "tolong"] {
        let path = format!("rbigint::<Impl>::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "not_rpython")),
            "missing upstream @not_rpython marker for {path}: {hints:?}"
        );
    }
    assert!(
        hints
            .get("rbigint::args_from_long")
            .is_some_and(|values| values.iter().any(|hint| hint == "not_rpython")),
        "missing upstream @not_rpython marker for rbigint::args_from_long: {hints:?}"
    );
    for name in [
        "digit",
        "widedigit",
        "uwidedigit",
        "udigit",
        "setdigit",
        "setdigit_udigit",
        "setdigit_widedigit",
        "setdigit_uwidedigit",
        "numdigits",
        "lshift",
        "lqshift",
        "_normalize",
    ] {
        let path = format!("rbigint::<Impl>::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "always_inline")),
            "missing upstream _always_inline_ marker for {path}: {hints:?}"
        );
    }
    for name in [
        "int_in_valid_range",
        "_load_unsigned_digit",
        "_bitcount64_ops",
        "_format_lowest_level_divmod_int_results",
    ] {
        let path = format!("rbigint::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "always_inline")),
            "missing upstream _always_inline_=True marker for {path}: {hints:?}"
        );
    }
    assert!(
        hints
            .get("rbigint::<Impl>::rshift")
            .is_some_and(|values| values.iter().any(|hint| hint == "always_inline_try")),
        "missing upstream _always_inline_='try' marker for rbigint::<Impl>::rshift: {hints:?}"
    );
    assert!(
        hints
            .get("rbigint::_AsDouble")
            .is_some_and(|values| values.iter().any(|hint| hint == "dont_look_inside")),
        "missing upstream @jit.dont_look_inside marker for rbigint::_AsDouble: {hints:?}"
    );
    for (path, effect) in [
        ("longobject::jit_bigint_from_i64", "elidable_or_memerror"),
        ("longobject::jit_bigint_eq", "elidable_cannot_raise"),
    ] {
        let values = hints
            .get(path)
            .unwrap_or_else(|| panic!("missing pointer-ABI wrapper hints for {path}"));
        assert!(values.iter().any(|hint| hint == "elidable"));
        assert!(
            values.iter().any(|hint| hint == effect),
            "{path} must carry {effect}, got {values:?}"
        );
    }
}

#[test]
fn rbigint_inherent_constructors_keep_their_owner_and_graph() {
    if !std::path::Path::new(OBJECT_LLBC).is_file() {
        eprintln!(
            "skipping: {OBJECT_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-object`"
        );
        return;
    }

    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
        &[llbc],
        HostStaticAddrs::default(),
        &["rbigint", "longobject"],
    )
    .expect("lower rbigint module");

    for name in [
        "new",
        "setdigit",
        "setdigit_udigit",
        "setdigit_widedigit",
        "setdigit_uwidedigit",
        "fromint",
        "fromrarith_int",
        "fromrarith_uint",
    ] {
        let function = program
            .functions
            .iter()
            .find(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
            })
            .unwrap_or_else(|| panic!("missing rbigint::RBigInt::{name} graph"));
        assert_eq!(function.module_path, "rbigint::<Impl>");
    }
    for name in [
        "_mask_digit",
        "_mask_udigit",
        "_mask_widedigit",
        "_mask_uwidedigit",
        "_store_digit",
        "_store_udigit",
        "_store_widedigit",
        "_store_uwidedigit",
        "digits_from_nonneg_long",
        "digits_from_nonneg_ulong",
        "args_from_rarith_int1",
        "args_from_rarith_int",
        "args_from_rarith_uint1",
        "args_from_rarith_uint",
        "_loghelper_ln",
        "_loghelper_log10",
        "_loghelper_log2",
        "_format_recursive_decimal",
        "_format_recursive_general",
        "_bitwise_and",
        "_bitwise_or",
        "_bitwise_xor",
        "_int_bitwise_and",
        "_int_bitwise_or",
        "_int_bitwise_xor",
    ] {
        assert!(
            program
                .functions
                .iter()
                .any(|function| function.name == name && function.module_path == "rbigint"),
            "missing specialized upstream helper graph rbigint::{name}"
        );
    }

    let input_types = |name: &str, owner: Option<&str>| {
        let function = program
            .functions
            .iter()
            .find(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == owner
                    && function.module_path
                        == if owner.is_some() {
                            "rbigint::<Impl>"
                        } else {
                            "rbigint"
                        }
            })
            .unwrap_or_else(|| panic!("missing typed rbigint graph {owner:?}::{name}"));
        function
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Input { ty, .. } => Some(ty.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    for (name, expected) in [
        ("_mask_digit", ValueType::Int),
        ("_mask_udigit", ValueType::Unsigned),
        ("_mask_widedigit", ValueType::Int128),
        ("_mask_uwidedigit", ValueType::UInt128),
        ("_store_digit", ValueType::Int),
        ("_store_udigit", ValueType::Unsigned),
        ("_store_widedigit", ValueType::Int128),
        ("_store_uwidedigit", ValueType::UInt128),
        ("digits_from_nonneg_long", ValueType::Int),
        ("digits_from_nonneg_ulong", ValueType::Unsigned),
        ("args_from_rarith_int1", ValueType::Int),
        ("args_from_rarith_int", ValueType::Int),
        ("args_from_rarith_uint1", ValueType::Unsigned),
        ("args_from_rarith_uint", ValueType::Unsigned),
    ] {
        assert_eq!(
            input_types(name, None),
            vec![expected],
            "wrong @specialize.argtype graph carrier for rbigint::{name}"
        );
    }
    for (name, expected) in [
        ("setdigit", ValueType::Int),
        ("setdigit_udigit", ValueType::Unsigned),
        ("setdigit_widedigit", ValueType::Int128),
        ("setdigit_uwidedigit", ValueType::UInt128),
    ] {
        assert_eq!(
            input_types(name, Some("rbigint::RBigInt")).last(),
            Some(&expected),
            "wrong @specialize.argtype value carrier for RBigInt::{name}"
        );
        assert_eq!(
            input_types(name, Some("rbigint::RBigInt")).get(1),
            Some(&ValueType::Int),
            "RPython digit index must remain Signed for RBigInt::{name}"
        );
    }
    for name in ["digit", "widedigit", "uwidedigit", "udigit"] {
        assert_eq!(
            input_types(name, Some("rbigint::RBigInt")).last(),
            Some(&ValueType::Int),
            "RPython digit index must remain Signed for RBigInt::{name}"
        );
    }
    for name in ["lqshift", "rqshift"] {
        assert_eq!(
            input_types(name, Some("rbigint::RBigInt")).last(),
            Some(&ValueType::Int),
            "RPython quick-shift count must remain Signed for RBigInt::{name}"
        );
    }
    let from_list_types = input_types("from_list_n_bits", Some("rbigint::RBigInt"));
    assert!(
        !from_list_types.contains(&ValueType::Unsigned)
            && from_list_types.last() == Some(&ValueType::Int),
        "RPython from_list_n_bits length/bit-width inputs must remain Signed: \
         {from_list_types:?}"
    );
    for name in ["_loghelper_ln", "_loghelper_log10", "_loghelper_log2"] {
        let types = input_types(name, None);
        assert!(
            matches!(types.as_slice(), [ValueType::Ref(_)]),
            "specialized logarithm graph must carry only the RBigInt argument: {types:?}"
        );
    }
    for name in ["_bitwise_and", "_bitwise_or", "_bitwise_xor"] {
        let types = input_types(name, None);
        assert!(
            matches!(types.as_slice(), [ValueType::Ref(_), ValueType::Ref(_)]),
            "specialized bigint bitwise graph must not carry an operation discriminator: \
             {types:?}"
        );
    }
    for name in ["_int_bitwise_and", "_int_bitwise_or", "_int_bitwise_xor"] {
        let types = input_types(name, None);
        assert!(
            matches!(types.as_slice(), [ValueType::Ref(_), ValueType::Int]),
            "specialized int bitwise graph must not carry an operation discriminator: {types:?}"
        );
    }
    assert_eq!(
        input_types("_format_recursive_decimal", None),
        input_types("_format_recursive_general", None),
        "recursive formatting specializations must differ only in their fixed callable"
    );
    for name in [
        "_kmul_split",
        "_v_iadd",
        "_v_isub",
        "_v_lshift",
        "_v_rshift",
        "_extract_digits",
        "div2n1n",
        "div3n2n",
        "_full_digits_lshift_then_or",
        "_format_recursive_decimal",
        "_format_recursive_general",
        "_format_lowest_level_divmod_int_results",
        "get_cached_parts",
        "digits_max_for_base",
        "_decimalstr_to_bigint",
        "_str_to_int_big_w5pow",
        "_str_to_int_big_inner10",
        "_str_to_int_big_base10",
        "tobytes_int",
    ] {
        let types = input_types(name, None);
        assert!(
            !types.contains(&ValueType::Unsigned),
            "upstream logical size/index leaked as Rust Unsigned in rbigint::{name}: {types:?}"
        );
    }
    for old_runtime_dispatch in [
        "_loghelper",
        "_format_recursive",
        "_bitwise",
        "_int_bitwise",
    ] {
        assert!(
            !program.functions.iter().any(|function| {
                function.name == old_runtime_dispatch && function.module_path == "rbigint"
            }),
            "runtime-dispatch graph rbigint::{old_runtime_dispatch} survived @specialize.arg port"
        );
    }

    let x_divrem = program
        .functions
        .iter()
        .find(|function| function.name == "_x_divrem" && function.module_path == "rbigint")
        .expect("rbigint::_x_divrem wide-digit graph");
    let x_divrem_ops = x_divrem
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect::<Vec<_>>();
    assert!(
        x_divrem_ops.iter().any(|operation| matches!(
            &operation.kind,
            OpKind::BinOp {
                result_ty: majit_translate::model::ValueType::Int128
                    | majit_translate::model::ValueType::UInt128,
                ..
            }
        )),
        "_x_divrem must preserve 128-bit LONG_TYPE/ULONG_TYPE binop results"
    );
    for name in ["fromlong", "tolong"] {
        assert!(
            !program.functions.iter().any(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
            }),
            "upstream @not_rpython method RBigInt::{name} must not have a flow graph"
        );
    }
    assert!(
        !program
            .functions
            .iter()
            .any(|function| function.name == "args_from_long"),
        "upstream @not_rpython helper args_from_long must not have a flow graph"
    );

    let fromint = program
        .functions
        .iter()
        .find(|function| {
            function.name == "fromint"
                && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("fromint graph");
    let mut saw_mask = false;
    for block in &fromint.graph.blocks {
        for operation in &block.operations {
            match &operation.kind {
                OpKind::ConstInt(value) if *value == i64::MAX => {
                    saw_mask = true;
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } => {
                    assert_ne!(
                        segments.last().map(String::as_str),
                        Some("MASK"),
                        "a NamedConst read must fold instead of becoming a call"
                    );
                }
                _ => {}
            }
        }
    }
    assert!(
        saw_mask,
        "fromint must contain the folded 63-bit digit mask"
    );

    let constructor = program
        .functions
        .iter()
        .find(|function| {
            function.name == "new" && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("new graph");
    for block in &constructor.graph.blocks {
        for operation in &block.operations {
            if let OpKind::Call { target, .. } = &operation.kind {
                assert_ne!(
                    target
                        .path_segments()
                        .and_then(|parts| parts.last().copied()),
                    Some("is_ascii"),
                    "the translated constructor must not retain debug formatting machinery: \
                     {target:?}"
                );
            }
        }
    }

    let gt = program
        .functions
        .iter()
        .find(|function| {
            function.name == "gt" && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("RBigInt::gt graph");
    assert!(
        gt.graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|operation| matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments
                    == &["pyre_object", "longobject", "jit_bigint_lt"]
            )),
        "the local Method-form `other.lt(self)` must become a scalar residual"
    );

    let is_zero = program
        .functions
        .iter()
        .find(|function| {
            function.name == "is_zero"
                && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("RBigInt::is_zero graph");
    assert!(
        is_zero
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|operation| matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments
                    == &["pyre_object", "longobject", "jit_bigint_sign_i64"]
            )),
        "the local Method-form `get_sign()` must become a scalar residual"
    );

    for (name, forbidden_leaf) in [("int_pow", "pow"), ("rshift", "rqshift")] {
        let function = program
            .functions
            .iter()
            .find(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
            })
            .unwrap_or_else(|| panic!("missing rbigint::RBigInt::{name} graph"));
        for operation in function
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
        {
            if let OpKind::Call { target, .. } = &operation.kind {
                assert_ne!(
                    target
                        .path_segments()
                        .and_then(|parts| parts.last().copied()),
                    Some(forbidden_leaf),
                    "PyPy's specialized RBigInt::{name} body must not delegate to \
                     RBigInt::{forbidden_leaf}: {target:?}"
                );
            }
        }
    }
}

#[test]
fn dependent_crate_rbigint_identity_retargets_opaque_llbc_declaration() {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file() {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-interpreter`"
        );
        return;
    }

    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let hints = harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
    for path in [
        "objspace::descroperation::jit_bigint_add",
        "objspace::descroperation::jit_bigint_int_add",
        "objspace::descroperation::jit_bigint_int_sub",
        "objspace::descroperation::jit_bigint_int_mul",
        "objspace::descroperation::jit_bigint_int_and",
        "objspace::descroperation::jit_bigint_int_or",
        "objspace::descroperation::jit_bigint_int_xor",
        "objspace::descroperation::jit_bigint_shl",
        "objspace::descroperation::jit_bigint_pow_nomod",
        "objspace::descroperation::jit_bigint_div_floor",
        "objspace::descroperation::jit_bigint_mod_floor",
        "jit_compiler_bigint_to_rbigint",
    ] {
        let values = hints
            .get(path)
            .unwrap_or_else(|| panic!("missing pointer-ABI wrapper hints for {path}"));
        assert!(values.iter().any(|hint| hint == "elidable"));
        assert!(
            values.iter().any(|hint| hint == "elidable_or_memerror"),
            "{path} must be allocation-only elidable, got {values:?}"
        );
    }
    // rbigint.py `_make_int_comparison` only reads existing digits and
    // returns a bool.  These wrappers are plain @jit.elidable, not the
    // allocation-only arithmetic contract above.
    for path in [
        "objspace::descroperation::jit_bigint_int_eq",
        "objspace::descroperation::jit_bigint_int_ne",
        "objspace::descroperation::jit_bigint_int_lt",
        "objspace::descroperation::jit_bigint_int_le",
        "objspace::descroperation::jit_bigint_int_gt",
        "objspace::descroperation::jit_bigint_int_ge",
    ] {
        let values = hints
            .get(path)
            .unwrap_or_else(|| panic!("missing pointer-ABI wrapper hints for {path}"));
        assert!(values.iter().any(|hint| hint == "elidable"));
        assert!(
            !values.iter().any(|hint| hint == "elidable_or_memerror"),
            "{path} comparison must not advertise an allocation edge: {values:?}"
        );
    }
    for path in [
        "objspace::descroperation::jit_bigint_bit_length",
        "objspace::descroperation::jit_bigint_bit_count",
    ] {
        let values = hints
            .get(path)
            .unwrap_or_else(|| panic!("missing pointer-ABI wrapper hints for {path}"));
        assert!(values.iter().any(|hint| hint == "elidable"));
        assert!(
            !values.iter().any(|hint| hint == "elidable_cannot_raise"),
            "{path} must preserve rbigint ovfcheck's OverflowError edge, got {values:?}"
        );
    }
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
        &[llbc],
        HostStaticAddrs::default(),
        &["objspace::descroperation"],
    )
    .expect("lower descroperation module");
    let helper = program
        .functions
        .iter()
        .find(|function| {
            function.name == "bigint_add"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::bigint_add graph");

    let calls: Vec<Vec<String>> = helper
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } = &operation.kind
            {
                Some(segments.clone())
            } else {
                None
            }
        })
        .collect();
    let call_targets: Vec<String> = helper
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| {
            if let OpKind::Call { target, .. } = &operation.kind {
                Some(format!("{target:?}"))
            } else {
                None
            }
        })
        .collect();
    let residuals = calls
        .iter()
        .filter(|segments| {
            segments.as_slice()
                == [
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    "jit_bigint_add",
                ]
        })
        .count();
    assert_eq!(
        residuals, 1,
        "the dependent-crate opaque declaration must retain RBigInt identity; \
         calls={calls:?}; targets={call_targets:?}"
    );

    for (caller_name, residual_name) in [
        ("long_add", "jit_bigint_int_add"),
        ("long_sub", "jit_bigint_int_sub"),
        ("long_mul", "jit_bigint_int_mul"),
        ("long_bitand", "jit_bigint_int_and"),
        ("long_bitor", "jit_bigint_int_or"),
        ("long_bitxor", "jit_bigint_int_xor"),
    ] {
        let caller = program
            .functions
            .iter()
            .find(|function| {
                function.name == caller_name
                    && function.module_path.ends_with("objspace::descroperation")
            })
            .unwrap_or_else(|| panic!("descroperation::{caller_name} graph"));
        let calls: Vec<Vec<String>> = caller
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect();
        assert!(
            calls
                .iter()
                .any(|segments| segments.last().is_some_and(|leaf| leaf == residual_name)),
            "{caller_name} must preserve PyPy's mixed-long/int {residual_name} call: {calls:?}"
        );
        assert!(
            !calls.iter().any(|segments| {
                segments.last().is_some_and(|leaf| {
                    matches!(
                        leaf.as_str(),
                        "int_add" | "int_sub" | "int_mul" | "int_and_" | "int_or_" | "int_xor"
                    )
                })
            }),
            "{caller_name} retained an untranslated RBigInt::int_* call: {calls:?}"
        );
    }

    let mixed_compare = program
        .functions
        .iter()
        .find(|function| {
            function.name == "long_int_compare"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::long_int_compare graph");
    let mixed_compare_calls: Vec<Vec<String>> = mixed_compare
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    for residual_name in [
        "jit_bigint_int_eq",
        "jit_bigint_int_ne",
        "jit_bigint_int_lt",
        "jit_bigint_int_le",
        "jit_bigint_int_gt",
        "jit_bigint_int_ge",
    ] {
        assert!(
            mixed_compare_calls
                .iter()
                .any(|segments| segments.last().is_some_and(|leaf| leaf == residual_name)),
            "long_int_compare must target {residual_name}: {mixed_compare_calls:?}"
        );
    }
    assert!(
        !mixed_compare_calls.iter().any(|segments| {
            segments.last().is_some_and(|leaf| {
                matches!(
                    leaf.as_str(),
                    "int_eq" | "int_ne" | "int_lt" | "int_le" | "int_gt" | "int_ge"
                )
            })
        }),
        "long_int_compare retained an untranslated RBigInt::int_* call: \
         {mixed_compare_calls:?}"
    );

    let long_pow = program
        .functions
        .iter()
        .find(|function| {
            function.name == "long_pow"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::long_pow graph");
    let pow_calls: Vec<Vec<String>> = long_pow
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(
        pow_calls.iter().any(|segments| segments
            == &[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_pow_nomod",
            ]),
        "long_pow must erase Rust Result to the elidable pointer ABI: {pow_calls:?}"
    );
    assert!(
        !pow_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "bigint_pow_nomod")),
        "the host Result wrapper must not survive in the translated graph: {pow_calls:?}"
    );

    let long_lshift = program
        .functions
        .iter()
        .find(|function| {
            function.name == "long_lshift"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::long_lshift graph");
    let lshift_calls: Vec<Vec<String>> = long_lshift
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(
        lshift_calls.iter().any(|segments| segments
            == &[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_lshift_count",
            ]),
        "long_lshift must erase Rust Result to the elidable pointer ABI: {lshift_calls:?}"
    );
    assert!(
        !lshift_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "bigint_lshift_count")),
        "the host Result wrapper must not survive in the translated graph: {lshift_calls:?}"
    );

    for (caller_name, residual_name) in [
        ("long_floordiv", "jit_bigint_div_floor"),
        ("long_mod", "jit_bigint_mod_floor"),
    ] {
        let caller = program
            .functions
            .iter()
            .find(|function| {
                function.name == caller_name
                    && function.module_path.ends_with("objspace::descroperation")
            })
            .unwrap_or_else(|| panic!("descroperation::{caller_name} graph"));
        let calls: Vec<Vec<String>> = caller
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect();
        assert!(
            calls
                .iter()
                .any(|segments| { segments.last().is_some_and(|leaf| leaf == residual_name) }),
            "{caller_name} must target {residual_name}: {calls:?}"
        );
    }
}

#[test]
fn rbigint_operator_calls_retarget_to_gc_reference_residuals() {
    if !std::path::Path::new(OBJECT_LLBC).is_file() {
        eprintln!(
            "skipping: {OBJECT_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-object`"
        );
        return;
    }

    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
        &[llbc],
        HostStaticAddrs::default(),
        &["longobject"],
    )
    .expect("lower longobject module");
    let wrapper = program
        .functions
        .iter()
        .find(|function| function.name == "jit_bigint_add" && function.module_path == "longobject")
        .expect("longobject::jit_bigint_add graph");

    let mut residual_calls = 0;
    for block in &wrapper.graph.blocks {
        for operation in &block.operations {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } = &operation.kind
            {
                assert!(
                    !matches!(segments.as_slice(), [.., owner, leaf]
                        if owner == "<Impl>" && leaf == "add"),
                    "the Rust by-value RBigInt trait shim must not enter the JIT graph: \
                     {segments:?}"
                );
                if segments
                    == &[
                        "pyre_interpreter",
                        "objspace",
                        "descroperation",
                        "jit_bigint_add",
                    ]
                {
                    residual_calls += 1;
                }
            }
        }
    }
    assert_eq!(
        residual_calls, 1,
        "RBigInt addition must be one GC-reference residual call"
    );

    let constructor_caller = program
        .functions
        .iter()
        .find(|function| function.name == "w_long_from_i64" && function.module_path == "longobject")
        .expect("longobject::w_long_from_i64 graph");
    let constructor_residuals = constructor_caller
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|operation| {
            matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments
                    == &[
                        "pyre_object",
                        "longobject",
                        "jit_bigint_from_i64",
                    ]
            )
        })
        .count();
    assert_eq!(
        constructor_residuals, 1,
        "RBigInt::from(i64) must return one GC reference through its residual"
    );
}
