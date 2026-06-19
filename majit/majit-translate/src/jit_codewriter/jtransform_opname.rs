//! Opname-dispatch transducer for the codewriter convergence ("Spine B").
//!
//! The production codewriter consumes the rich-`OpKind`
//! `crate::model::FunctionGraph` (`jtransform.rs`'s `Transformer::transform`
//! dispatches on `OpKind::FieldRead`/`OpKind::Call`/…).  The rtyper, however,
//! already lowers the *flowspace* graph to upstream-shaped low-level
//! `SpaceOperation`s (`getfield`/`setfield`/`getarrayitem`/`malloc_varsize`/
//! `int_add`/…) in place, and certain helper graphs (the `ll_str*` family
//! built by `lltypesystem/rstr.rs`) are born ONLY in that opname form — they
//! have no rich-`OpKind` twin and are discarded today.
//!
//! [`lower_graph`] is the convergence transducer: it consumes such a
//! `crate::flowspace::model::FunctionGraph` and emits an equivalent
//! `crate::model::FunctionGraph` (rich `OpKind`), which then re-enters the
//! EXISTING flatten/regalloc/assembler tail unchanged
//! (`CodeWriter::finalize_rewritten_graph_to_jitcode`).  This is the port of
//! `jtransform.py`'s `_rewrite_ops[op.opname]` dispatch
//! (`jtransform.py:238`), reading each `Variable.concretetype` directly (the
//! upstream `getkind(v.concretetype)` path) rather than through the
//! `value_to_var` bridge the rich-`OpKind` spine uses.
//!
//! Coexistence: a graph's ops are either all rich-`OpKind` (Spine A) or all
//! opname (Spine B); the two never mix within one graph.  The drain loop
//! routes a path to this module only when
//! `CallControl::take_opname_graph` returns its registered flowspace graph.
//! Until a helper is registered via
//! `CallControl::register_opname_helper_graph`, this module is dead code.
//!
//! ## String-helper fusion (S1 scope)
//!
//! The string-repr family stores its character data as an `Array(Char)` /
//! `Array(UniChar)` *inline* in the GC `STR` / `UNICODE` struct, so a
//! `getsubstruct(s, "chars")` yields an interior pointer with no standalone
//! runtime object.  The blackhole interpreter has no interior-pointer model;
//! instead it carries by-name handlers that take the **string object** plus
//! register-shaped operands: `strlen(s)`, `strgetitem(s, i)`,
//! `strsetitem(s, i, c)`, `newstr(n)` (and the `unicode*` / `newunicode`
//! peers).  The transducer therefore *fuses* the `getsubstruct` + array op
//! pair back into the string opcode — `getsubstruct` itself emits nothing and
//! records `chars_array_var → string_var`; a following `getarraysize` /
//! `getarrayitem` / `setarrayitem` on that array re-references the recorded
//! string operand.  This mirrors upstream `s.chars[i]`, where the chars
//! substruct is re-derived at each access from the string in hand.
//!
//! The fusion is **block-local**: it relies on the `getsubstruct` and its
//! consuming array op living in the same block, with the string operand
//! available there.  A helper that hoists the substruct out of a loop and
//! threads the *interior chars pointer* through block Phi inputargs (rather
//! than threading the string itself) cannot be fused this way — the string
//! origin is lost across the Phi.  Such graphs are out of scope for this
//! slice; the transducer fail-loud `expect`s a recorded alias rather than
//! silently miscompiling.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::Variable as FVar;
use crate::flowspace::model::{
    Block as FlowBlock, ConstValue, FunctionGraph as FlowGraph, Hlvalue,
};
use crate::model::{BlockId, ExitCase, ExitSwitch, Link, LinkArg, OpKind, ValueType};
use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};

/// Block identity key — flowspace blocks are `Rc<RefCell<Block>>` with no
/// inherent id, so identity is the allocation address (RPython compares
/// `Block` objects by Python identity; `iterblocks` already keys its
/// visited-set on `Rc::as_ptr`).
type FlowBlockKey = *const RefCell<FlowBlock>;

/// The blackhole opcode family for a given string-element width.  `str*`
/// handlers operate on `Array(Char)`-backed `STR`; `unicode*` /
/// `newunicode` on `Array(UniChar)`-backed `UNICODE`.
struct StrFamily {
    /// `strlen` / `unicodelen` — array length of the inline chars array.
    len: &'static str,
    /// `strgetitem` / `unicodegetitem` — read one element.
    getitem: &'static str,
    /// `strsetitem` / `unicodesetitem` — write one element (void).
    setitem: &'static str,
    /// `newstr` / `newunicode` — variable-size allocation by length.
    alloc: &'static str,
}

impl StrFamily {
    const STR: StrFamily = StrFamily {
        len: "strlen",
        getitem: "strgetitem",
        setitem: "strsetitem",
        alloc: "newstr",
    };
    const UNICODE: StrFamily = StrFamily {
        len: "unicodelen",
        getitem: "unicodegetitem",
        setitem: "unicodesetitem",
        alloc: "newunicode",
    };
}

/// Lower an rtyper low-level helper graph (opname `SpaceOperation`s) to an
/// equivalent rich-`OpKind` `crate::model::FunctionGraph`.
///
/// The flowspace `Variable`s are reused verbatim as the model graph's
/// operands and block inputargs — they are the same
/// `crate::flowspace::model::Variable` type and carry their `concretetype`
/// cell through unchanged, so no value-bridge round-trip is needed.
pub fn lower_graph(graph: &FlowGraph) -> crate::model::FunctionGraph {
    let mut out = crate::model::FunctionGraph::new(graph.name.clone());
    let family = detect_family(graph);

    // Map each flowspace block to a model `BlockId`.  The three canonical
    // blocks (`startblock`/`returnblock`/`exceptblock`) are pre-seeded onto
    // the model graph's own canonical ids so links into them resolve.
    let mut block_map: HashMap<FlowBlockKey, BlockId> = HashMap::new();
    block_map.insert(Rc::as_ptr(&graph.startblock), out.startblock);
    block_map.insert(Rc::as_ptr(&graph.returnblock), out.returnblock);
    block_map.insert(Rc::as_ptr(&graph.exceptblock), out.exceptblock);

    let flow_blocks = graph.iterblocks();

    // Pass 1 — allocate model blocks for every reachable block and copy each
    // block's inputarg `Variable`s across (the canonical blocks already
    // exist; interior blocks get fresh ids).
    for fb in &flow_blocks {
        let key = Rc::as_ptr(fb);
        let id = *block_map.entry(key).or_insert_with(|| out.create_block());
        let inputargs: Vec<FVar> = fb
            .borrow()
            .inputargs
            .iter()
            .filter_map(hlvalue_as_var)
            .collect();
        out.block_mut(id).inputargs = inputargs;
    }
    // The return/except blocks may not appear in `iterblocks` (e.g. a graph
    // with no raising edge never reaches `exceptblock`); set their inputargs
    // from the flowspace canonical blocks so the model return value carries
    // the helper's concretetype.
    let return_inputargs: Vec<FVar> = graph
        .returnblock
        .borrow()
        .inputargs
        .iter()
        .filter_map(hlvalue_as_var)
        .collect();
    let return_id = out.returnblock;
    out.block_mut(return_id).inputargs = return_inputargs;

    // Pass 2 — transduce each block's operations + control flow.
    for fb in &flow_blocks {
        let id = block_map[&Rc::as_ptr(fb)];
        let fb_ref = fb.borrow();

        // `getsubstruct` aliases are block-local: a chars-array Variable maps
        // back to the string Variable it was derived from in this block.
        let mut chars_alias: HashMap<FVar, FVar> = HashMap::new();
        for op in &fb_ref.operations {
            transduce_op(&mut out, id, op, &mut chars_alias, &family);
        }

        let exitswitch = if fb_ref.canraise() {
            Some(ExitSwitch::LastException)
        } else {
            match &fb_ref.exitswitch {
                Some(Hlvalue::Variable(v)) => Some(ExitSwitch::Value(v.clone())),
                _ => None,
            }
        };

        let exits: Vec<Link> = fb_ref
            .exits
            .iter()
            .map(|link_ref| {
                let link = link_ref.borrow();
                let target = link
                    .target
                    .as_ref()
                    .expect("opname-spine link has no target block");
                let target_id = block_map[&Rc::as_ptr(target)];
                let args: Vec<LinkArg> = link
                    .args
                    .iter()
                    .map(|arg| {
                        linkarg_from_hlvalue(
                            arg.as_ref().expect("opname-spine link arg is undefined"),
                        )
                    })
                    .collect();
                let exitcase = match &link.exitcase {
                    Some(Hlvalue::Constant(c)) => exitcase_from_const(&c.value),
                    _ => None,
                };
                Link::new_mixed(args, target_id, exitcase)
            })
            .collect();

        drop(fb_ref);
        if exitswitch.is_some() || !exits.is_empty() {
            out.set_control_flow_metadata(id, exitswitch, exits);
        }
    }

    out
}

/// Transduce a single flowspace `SpaceOperation` into the model graph at
/// `block`, materialising any constant operands as `ConstInt`/`ConstBool`
/// ops and fusing `getsubstruct`-derived array accesses into the string
/// blackhole opcodes.
fn transduce_op(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
    chars_alias: &mut HashMap<FVar, FVar>,
    family: &StrFamily,
) {
    match op.opname.as_str() {
        // `getsubstruct(s, "chars")` — interior pointer with no runtime
        // object; record the alias and emit nothing.  The string operand is
        // always a Variable in the helper graphs.
        "getsubstruct" => {
            let string_var = expect_var(&op.args[0]);
            let chars_var = expect_var(&op.result);
            chars_alias.insert(chars_var, string_var);
        }
        // `len(s.chars)` → `strlen(s)` / `unicodelen(s)`.
        "getarraysize" => {
            let string_var = resolve_string(chars_alias, &op.args[0]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: family.len.to_string(),
                    args: vec![string_var],
                },
                result,
            );
        }
        // `s.chars[i]` → `strgetitem(s, i)` / `unicodegetitem(s, i)`.
        "getarrayitem" => {
            let string_var = resolve_string(chars_alias, &op.args[0]);
            let index = materialize(out, block, &op.args[1]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: family.getitem.to_string(),
                    args: vec![string_var, index],
                },
                result,
            );
        }
        // `s.chars[i] = c` → `strsetitem(s, i, c)` / `unicodesetitem(...)`.
        // Void result.
        "setarrayitem" => {
            let string_var = resolve_string(chars_alias, &op.args[0]);
            let index = materialize(out, block, &op.args[1]);
            let value = materialize(out, block, &op.args[2]);
            out.push_op_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: family.setitem.to_string(),
                    args: vec![string_var, index, value],
                },
                false,
            );
        }
        // `malloc_varsize(STR, gc, n)` → `newstr(n)` / `newunicode(n)`.  The
        // struct lltype and gc-flavor operands carry no runtime value for the
        // blackhole allocator, which is keyed by the string family.
        "malloc_varsize" => {
            let size = materialize(out, block, &op.args[2]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: family.alloc.to_string(),
                    args: vec![size],
                },
                result,
            );
        }
        // `copystrcontent(src, dst, srcstart, dststart, length)` /
        // `copyunicodecontent(...)` — bulk char copy.  Both string operands
        // are whole objects (not chars-array interior pointers), so they
        // bypass the `chars_alias` map.  Void result.
        "copystrcontent" | "copyunicodecontent" => {
            let src = expect_var(&op.args[0]);
            let dst = expect_var(&op.args[1]);
            let srcstart = materialize(out, block, &op.args[2]);
            let dststart = materialize(out, block, &op.args[3]);
            let length = materialize(out, block, &op.args[4]);
            out.push_op_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: op.opname.clone(),
                    args: vec![src, dst, srcstart, dststart, length],
                },
                false,
            );
        }
        // Integer arithmetic / comparison — `int_add`/`int_lt`/… → `BinOp`
        // with the bare op name (the assembler re-prefixes `int_`).  Constant
        // operands materialise as `ConstInt`/`ConstBool` since `BinOp` takes
        // two Variables.
        name if name.starts_with("int_") => {
            let bare = name.strip_prefix("int_").unwrap().to_string();
            let lhs = materialize(out, block, &op.args[0]);
            let rhs = materialize(out, block, &op.args[1]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::BinOp {
                    op: bare,
                    lhs,
                    rhs,
                    result_ty,
                },
                result,
            );
        }
        other => panic!("jtransform_opname::lower_graph: unsupported opname {other:?}"),
    }
}

/// Resolve the string Variable backing a chars-array operand via the
/// block-local `getsubstruct` alias map.  Fail-loud if the array was not
/// produced by a same-block `getsubstruct` (e.g. threaded across a Phi —
/// out of scope for this slice).
fn resolve_string(chars_alias: &HashMap<FVar, FVar>, arr: &Hlvalue) -> FVar {
    let arr_var = expect_var(arr);
    chars_alias
        .get(&arr_var)
        .cloned()
        .expect("array operand was not a same-block getsubstruct(\"chars\") result")
}

/// Map a flowspace operand to a model operand `Variable`, materialising
/// constants into `ConstInt`/`ConstBool` ops pushed ahead of the consumer.
fn materialize(out: &mut crate::model::FunctionGraph, block: BlockId, hlv: &Hlvalue) -> FVar {
    match hlv {
        Hlvalue::Variable(v) => v.clone(),
        Hlvalue::Constant(c) => {
            let (kind, lltype) = match &c.value {
                ConstValue::Int(n) => (OpKind::ConstInt(*n), LowLevelType::Signed),
                ConstValue::Bool(b) => (OpKind::ConstBool(*b), LowLevelType::Bool),
                other => panic!(
                    "jtransform_opname::lower_graph: cannot materialise constant operand {other:?}"
                ),
            };
            let result = out
                .push_op_var(block, kind, true)
                .expect("ConstInt/ConstBool op produces a result var");
            // `push_op_var` mints the result with `ConcreteType::Unknown`;
            // stamp the int-bank kind so regalloc colours it.
            result.set_concretetype(Some(lltype));
            result
        }
    }
}

/// Map a flowspace link arg (`Variable` or `Constant`) to the model
/// `LinkArg` — model links carry mixed var/const args directly.
fn linkarg_from_hlvalue(hlv: &Hlvalue) -> LinkArg {
    match hlv {
        Hlvalue::Variable(v) => LinkArg::Value(v.clone()),
        Hlvalue::Constant(c) => LinkArg::Const(c.clone()),
    }
}

/// Map a flowspace exitcase constant to the model `ExitCase`.
fn exitcase_from_const(value: &ConstValue) -> Option<ExitCase> {
    match value {
        ConstValue::Bool(b) => Some(ExitCase::Bool(*b)),
        other => Some(ExitCase::Const(other.clone())),
    }
}

/// `Hlvalue::Variable` → its `Variable`; `None` for a constant.
fn hlvalue_as_var(hlv: &Hlvalue) -> Option<FVar> {
    match hlv {
        Hlvalue::Variable(v) => Some(v.clone()),
        Hlvalue::Constant(_) => None,
    }
}

/// Expect an `Hlvalue` to be a `Variable` (the position is always a Variable
/// in the helper graphs the transducer accepts).
fn expect_var(hlv: &Hlvalue) -> FVar {
    match hlv {
        Hlvalue::Variable(v) => v.clone(),
        Hlvalue::Constant(c) => {
            panic!("jtransform_opname::lower_graph: expected a Variable, found constant {c:?}")
        }
    }
}

/// Map a `Variable`'s `concretetype` to the model `ValueType` for `BinOp`'s
/// `result_ty` — the `getkind`-collapsed kind space (`Char`/`Bool`/`Signed`
/// all land in the int bank).
fn value_type_of(var: &FVar) -> ValueType {
    use crate::model::ConcreteType;
    match crate::model::FunctionGraph::concretetype_of(var) {
        ConcreteType::Signed => ValueType::Int,
        ConcreteType::GcRef => ValueType::Ref(None),
        ConcreteType::Float => ValueType::Float,
        ConcreteType::Void => ValueType::Void,
        ConcreteType::Unknown => ValueType::Int,
    }
}

/// A helper graph operates on a single string width, so the family is a
/// graph-wide property: `unicode*` if any operand carries an
/// `Array(UniChar)` pointer, else `str*`.
fn detect_family(graph: &FlowGraph) -> StrFamily {
    let is_unicode = graph.iterblocks().iter().any(|block| {
        block.borrow().operations.iter().any(|op| {
            op.args
                .iter()
                .chain(std::iter::once(&op.result))
                .any(|hlv| matches!(hlv, Hlvalue::Variable(v) if is_unichar_array_ptr(v)))
        })
    });
    if is_unicode {
        StrFamily::UNICODE
    } else {
        StrFamily::STR
    }
}

/// Whether a `Variable`'s `concretetype` is `Ptr(Array(UniChar))` — the
/// chars-array pointer of a `UNICODE` object.
fn is_unichar_array_ptr(var: &FVar) -> bool {
    match var.concretetype() {
        Some(LowLevelType::Ptr(ptr)) => {
            matches!(&ptr.TO, PtrTarget::Array(arr) if matches!(arr.OF, LowLevelType::UniChar))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::lower_graph;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, FunctionGraph as FlowGraph, Hlvalue, Link, SpaceOperation,
    };
    use crate::model::{ExitCase, ExitSwitch, OpKind};
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::lltypesystem::rstr::{
        STRPTR, chars_array_ptr_lltype_from_strptr, struct_lltype_from_strptr,
    };
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};
    use crate::translator::rtyper::rtyper::{constant_with_lltype, variable_with_lltype};

    fn chars_field() -> Hlvalue {
        constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void)
    }
    fn signed_const(n: i64) -> Hlvalue {
        constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed)
    }
    fn bool_const(b: bool) -> Hlvalue {
        constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool)
    }

    /// Build a single-string-type opname helper graph in the fusion-friendly
    /// shape: the chars substruct is re-derived *block-locally* from a
    /// threaded `STR` operand (never threaded as an interior pointer across a
    /// Phi).  Exercises the full S1 opname set:
    ///
    /// ```text
    /// start(s):
    ///     chars  = getsubstruct(s, "chars")
    ///     len    = getarraysize(chars)
    ///     newstr = malloc_varsize(STR, gc, len)
    ///     cond   = int_lt(0, len)
    ///     if cond -> copy(s, newstr) else -> return(newstr)
    /// copy(s, newstr):
    ///     chars_s  = getsubstruct(s, "chars")
    ///     c        = getarrayitem(chars_s, 0)
    ///     newchars = getsubstruct(newstr, "chars")
    ///     setarrayitem(newchars, 0, c)
    ///     -> return(newstr)
    /// ```
    fn build_fusable_str_helper() -> FlowGraph {
        let strptr = STRPTR.clone();
        let chars_ptr = chars_array_ptr_lltype_from_strptr(&strptr).expect("chars ptr lltype");
        let struct_lltype = struct_lltype_from_strptr(&strptr).expect("struct lltype");

        let s = variable_with_lltype("s", strptr.clone());
        let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
        let return_var = variable_with_lltype("result", strptr.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_fusable_str_helper",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );

        // copy block threads the STRING pointers (s, newstr), NOT interior
        // chars pointers — so each consumer re-derives its substruct locally.
        let s_c = variable_with_lltype("s", strptr.clone());
        let newstr_c = variable_with_lltype("newstr", strptr.clone());
        let copy_block = Block::shared(vec![
            Hlvalue::Variable(s_c.clone()),
            Hlvalue::Variable(newstr_c.clone()),
        ]);

        // ---- startblock.
        let chars = variable_with_lltype("chars", chars_ptr.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s.clone()), chars_field()],
            Hlvalue::Variable(chars.clone()),
        ));
        let len = variable_with_lltype("len", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars)],
            Hlvalue::Variable(len.clone()),
        ));
        let newstr = variable_with_lltype("newstr", strptr.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(struct_lltype),
                gc_flavor_const().expect("gc flavor const"),
                Hlvalue::Variable(len.clone()),
            ],
            Hlvalue::Variable(newstr.clone()),
        ));
        let cond = variable_with_lltype("cond", LowLevelType::Bool);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![signed_const(0), Hlvalue::Variable(len)],
            Hlvalue::Variable(cond.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(s), Hlvalue::Variable(newstr.clone())],
                Some(copy_block.clone()),
                Some(bool_const(true)),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(newstr)],
                Some(graph.returnblock.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        ]);

        // ---- copy block.
        let chars_s = variable_with_lltype("chars_s", chars_ptr.clone());
        copy_block.borrow_mut().operations.push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s_c), chars_field()],
            Hlvalue::Variable(chars_s.clone()),
        ));
        let c = variable_with_lltype("c", LowLevelType::Char);
        copy_block.borrow_mut().operations.push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(chars_s), signed_const(0)],
            Hlvalue::Variable(c.clone()),
        ));
        let newchars = variable_with_lltype("newchars", chars_ptr);
        copy_block.borrow_mut().operations.push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(newstr_c.clone()), chars_field()],
            Hlvalue::Variable(newchars.clone()),
        ));
        let set = variable_with_lltype("set", LowLevelType::Void);
        copy_block.borrow_mut().operations.push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(newchars),
                signed_const(0),
                Hlvalue::Variable(c),
            ],
            Hlvalue::Variable(set),
        ));
        copy_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(newstr_c)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        graph
    }

    #[test]
    fn lower_graph_fuses_str_helper_to_blackhole_opcodes() {
        let flow = build_fusable_str_helper();
        let model = lower_graph(&flow);

        // start(0) / return(1) / except(2) / copy(3).
        assert_eq!(model.blocks.len(), 4);

        let mut blackhole = Vec::new();
        let mut binops = Vec::new();
        let mut const_ints = 0;
        let mut residual = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::LoweredBlackholeOp { opname, .. } => blackhole.push(opname.clone()),
                    OpKind::BinOp { op, .. } => binops.push(op.clone()),
                    OpKind::ConstInt(_) => const_ints += 1,
                    other => residual.push(format!("{other:?}")),
                }
            }
        }
        blackhole.sort();

        // `getsubstruct` emits nothing; `getarraysize`/`getarrayitem`/
        // `setarrayitem`/`malloc_varsize` fuse to the string blackhole ops.
        assert_eq!(
            blackhole,
            vec!["newstr", "strgetitem", "strlen", "strsetitem"]
        );
        // `int_lt` lowers to a bare-named `BinOp`.
        assert_eq!(binops, vec!["lt"]);
        // The constant `0` is materialised once per consumer: `int_lt`,
        // `getarrayitem`, `setarrayitem`.
        assert_eq!(const_ints, 3);
        assert!(residual.is_empty(), "unexpected residual ops: {residual:?}");
    }

    #[test]
    fn lower_graph_preserves_bool_branch_control_flow() {
        let flow = build_fusable_str_helper();
        let model = lower_graph(&flow);

        // startblock is a 2-way bool branch on the `int_lt` result.
        let start = model.block(model.startblock);
        assert!(matches!(start.exitswitch, Some(ExitSwitch::Value(_))));
        assert_eq!(start.exits.len(), 2);
        let cases: Vec<&Option<ExitCase>> = start.exits.iter().map(|l| &l.exitcase).collect();
        assert!(cases.contains(&&Some(ExitCase::Bool(true))));
        assert!(cases.contains(&&Some(ExitCase::Bool(false))));

        // The true branch targets the copy block, which holds the read/write
        // string ops; the false branch returns directly.
        let true_link = start
            .exits
            .iter()
            .find(|l| l.exitcase == Some(ExitCase::Bool(true)))
            .expect("true exit");
        let copy_block = model.block(true_link.target);
        let copy_opnames: Vec<&str> = copy_block
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::LoweredBlackholeOp { opname, .. } => Some(opname.as_str()),
                _ => None,
            })
            .collect();
        assert!(copy_opnames.contains(&"strgetitem"));
        assert!(copy_opnames.contains(&"strsetitem"));

        let false_link = start
            .exits
            .iter()
            .find(|l| l.exitcase == Some(ExitCase::Bool(false)))
            .expect("false exit");
        assert_eq!(false_link.target, model.returnblock);
    }

    /// End-to-end: register the opname helper through Spine B and drive the
    /// drain loop, proving the transduced graph survives the shared
    /// regalloc/flatten/assemble tail and commits a non-empty `JitCode` body.
    /// This is the test that actually exercises the `LoweredBlackholeOp`
    /// assembler encode + dynamic byte assignment.
    #[test]
    fn lower_graph_drains_to_a_jitcode_body() {
        use crate::jit_codewriter::call::CallControl;
        use crate::jit_codewriter::codewriter::CodeWriter;
        use crate::jit_codewriter::jtransform::GraphTransformConfig;
        use crate::parse::CallPath;

        let flow = build_fusable_str_helper();
        let path = CallPath::from_segments(["ll_test_fusable_str_helper"]);

        let mut callcontrol = CallControl::new();
        let jitcode = callcontrol.register_opname_helper_graph(path, flow);
        assert!(jitcode.try_body().is_none(), "shell starts bodyless");

        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut callcontrol, &GraphTransformConfig::default());

        let body = jitcode
            .try_body()
            .expect("Spine-B drain commits a jitcode body");
        assert!(!body.code.is_empty(), "assembled bytecode is non-empty");
        assert_eq!(jitcode.try_index(), Some(0));
    }

    /// End-to-end: the restructured production `ll_strconcat` helper lowers
    /// cleanly through Spine B.  The two `copystrcontent` ops become void
    /// `LoweredBlackholeOp`s, the source-length reads fuse to `strlen`, and
    /// the cross-Phi `resolve_string` fail-loud is never reached because no
    /// per-char loop (strgetitem/strsetitem) survives.
    #[test]
    fn lower_graph_lowers_strconcat_helper_to_copystrcontent() {
        use crate::translator::rtyper::lltypesystem::rstr::{
            STRPTR, build_ll_strconcat_helper_graph,
        };

        let helper = build_ll_strconcat_helper_graph("ll_strconcat", STRPTR.clone())
            .expect("build strconcat helper");
        let flow = helper.graph.borrow();
        let model = lower_graph(&flow);

        let mut blackhole = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                if let OpKind::LoweredBlackholeOp { opname, args } = &op.kind {
                    blackhole.push(opname.clone());
                    if opname == "copystrcontent" {
                        assert_eq!(args.len(), 5, "copystrcontent has 5 operands");
                        assert!(op.result.is_none(), "copystrcontent is void");
                    }
                }
            }
        }

        // Two source-length reads (strlen) + one alloc (newstr) + two copies.
        assert_eq!(
            blackhole.iter().filter(|n| *n == "strlen").count(),
            2,
            "two strlen source-length reads"
        );
        assert_eq!(blackhole.iter().filter(|n| *n == "newstr").count(), 1);
        assert_eq!(
            blackhole.iter().filter(|n| *n == "copystrcontent").count(),
            2,
            "two copystrcontent bulk copies"
        );
        // No per-char string ops: the loop is gone, so resolve_string never runs.
        assert!(
            !blackhole
                .iter()
                .any(|n| n == "strgetitem" || n == "strsetitem"),
            "no per-char string ops survive: {blackhole:?}"
        );

        // start forwards unconditionally into the returnblock.
        let start = model.block(model.startblock);
        assert!(start.exitswitch.is_none(), "start exit is unconditional");
        assert_eq!(start.exits.len(), 1);
        assert_eq!(start.exits[0].target, model.returnblock);
    }
}
