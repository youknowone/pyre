//! `l[start:]` on a GC array in a graph the rtyper never lifted.
//!
//! The front spells a Rust `&slice[start..]` as the deferred marker call
//! `__getslice_rangefrom(slice, start)` (`front/slice_index.rs`).  A lifted
//! graph meets the rtyper, whose `rtype_getslice` (`rlist.py`) replaces the
//! `getslice` with `gendirectcall(ll_listslice_startonly, RESLIST, l, start)`
//! and mints that helper graph (`translator/rtyper/rlist.rs`).  A graph that
//! stays on the rich-`OpKind` spine never meets the rtyper, so the marker
//! survived to the codewriter as a call with no graph — a symbolic residual
//! no host symbol backs, and the one wall between a builtin gateway body and
//! its trace whenever the body passes `&args[1..]` on.
//!
//! This module gives that spine the same answer: [`listslice_startonly_path`]
//! mints an `ll_listslice_startonly` graph in the rich model for the array's
//! item kind and registers it as an ordinary callee, and `jtransform`
//! redirects the marker call to it.  The graph is `rlist.py`'s helper
//! line for line:
//!
//! ```python
//! def ll_listslice_startonly(RESLIST, l1, start):
//!     len1 = l1.ll_length()
//!     newlength = len1 - start
//!     l = RESLIST.ll_newlist(newlength)
//!     ll_arraycopy(l1, l, start, 0, newlength)
//!     return l
//! ```
//!
//! with `ll_arraycopy` written out as the item loop, because the rich model
//! has no arraycopy residual.  The graph carries the `unroll_safe` hint: the
//! sliced array is a gateway's argument array, whose length is the arity of
//! the CALL being traced, so the loop's trip count is a trace constant and
//! unrolling it is what makes the copied items virtual.
//!
//! The array's item kind is not on the marker call — the front knows it, the
//! marker does not carry it.  It is recovered from another array operation on
//! the same base in the same graph ([`array_identity_of_base`]); a graph that
//! slices an array it never otherwise reads keeps today's residual.

use crate::codewriter::call::CallControl;
use crate::flowspace::model::Variable;
use crate::model::{FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};
use crate::parse::CallPath;

/// The leaf every minted helper shares, suffixed per array identity.
pub const LL_LISTSLICE_STARTONLY: &str = "ll_listslice_startonly";

/// The item kind and ARRAY identity of the GC array `base` names, read off an
/// `ArrayRead` / `ArrayWrite` / `ArrayLen` / `NewArrayClear` on the same
/// value anywhere in the graph.  `None` when the graph never touches it as
/// an array.
///
/// "The same value" is judged across block links, not by `Variable`
/// identity: the front threads a live value through every block as a fresh
/// phi `inputarg`, so the array a slice reads in one block and the array the
/// marker slices in a later block are distinct `Variable`s bound together
/// only by the links between them.
pub fn array_identity_of_base(
    graph: &FunctionGraph,
    base: &Variable,
) -> Option<(ValueType, Option<String>)> {
    let classes = LinkClasses::of(graph);
    let same = |v: &Variable| classes.same(v, base);
    let mut len_only: Option<Option<String>> = None;
    let mut object_slice_input = false;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        match &op.kind {
            OpKind::ArrayRead {
                base: b,
                item_ty,
                array_type_id,
                ..
            }
            | OpKind::ArrayWrite {
                base: b,
                item_ty,
                array_type_id,
                ..
            } if same(b) => return Some((item_ty.clone(), array_type_id.clone())),
            OpKind::NewArrayClear {
                item_ty,
                array_type_id,
                ..
            } if op.result.as_ref().is_some_and(same) => {
                return Some((item_ty.clone(), array_type_id.clone()));
            }
            OpKind::ArrayLen {
                base: b,
                array_type_id,
                ..
            } if len_only.is_none() && same(b) => len_only = Some(array_type_id.clone()),
            OpKind::Input {
                class_root: Some(root),
                ..
            } if root == OBJECT_REF_SLICE_CLASS_ROOT && op.result.as_ref().is_some_and(same) => {
                object_slice_input = true;
            }
            _ => {}
        }
    }
    // A length read alone names the ARRAY but not its item kind; the object
    // array is the one identity whose items are known from the name.
    match len_only {
        Some(Some(id)) if id == crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID => {
            Some((ValueType::Ref(None), Some(id)))
        }
        // A `&[PyObjectRef]` parameter names its own identity: the front
        // stamps the slice's class root even where no sibling array op
        // carries an id.  A thin-pointer element keeps the unnamed id —
        // the identity-less descr mint already sizes it right, and it is
        // the id the graph's own length reads use.
        _ if object_slice_input => Some((ValueType::Ref(None), None)),
        _ => None,
    }
}

/// The class root the front stamps on a `&[PyObjectRef]` input
/// (`front/mir.rs` field/param layout spelling).
const OBJECT_REF_SLICE_CLASS_ROOT: &str = "[*mut PyObject]";

/// Union-find over the graph's links: `link.args[i]` and
/// `target.inputargs[i]` carry one value.
pub(crate) struct LinkClasses {
    parent: std::collections::HashMap<Variable, Variable>,
}

impl LinkClasses {
    pub(crate) fn of(graph: &FunctionGraph) -> Self {
        let mut classes = Self {
            parent: std::collections::HashMap::new(),
        };
        for block in &graph.blocks {
            for link in &block.exits {
                let target = graph.block(link.target);
                for (arg, input) in link.args.iter().zip(&target.inputargs) {
                    if let LinkArg::Value(value) = arg {
                        classes.union(value, input);
                    }
                }
            }
        }
        classes
    }

    pub(crate) fn find(&self, var: &Variable) -> Variable {
        let mut cur = var.clone();
        while let Some(next) = self.parent.get(&cur) {
            if *next == cur {
                break;
            }
            cur = next.clone();
        }
        cur
    }

    fn union(&mut self, a: &Variable, b: &Variable) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }

    pub(crate) fn same(&self, a: &Variable, b: &Variable) -> bool {
        a == b || self.find(a) == self.find(b)
    }
}

/// The helper path for `(item_ty, array_type_id)`, minting and registering
/// the graph on first use.
pub fn listslice_startonly_path(
    cc: &mut CallControl,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> CallPath {
    let name = helper_name(item_ty, array_type_id);
    let path = CallPath::from_segments([name.as_str()]);
    if !cc.has_function_graph(&path) {
        let graph = build_ll_listslice_startonly_graph(&name, item_ty, array_type_id);
        cc.register_function_graph_with_hints(path.clone(), graph, vec!["unroll_safe".into()]);
        cc.add_candidate_graph(path.clone());
    }
    path
}

/// `ll_listslice_startonly__<item>[__<array identity>]`, the identity
/// reduced to identifier characters so the leaf stays a plain path segment.
fn helper_name(item_ty: &ValueType, array_type_id: Option<&str>) -> String {
    let item = match item_ty {
        ValueType::Ref(_) => "ref",
        ValueType::Float => "float",
        _ => "int",
    };
    let mut name = format!("{LL_LISTSLICE_STARTONLY}__{item}");
    if let Some(id) = array_type_id {
        name.push_str("__");
        name.extend(
            id.chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }),
        );
    }
    name
}

/// `rlist.py ll_listslice_startonly` in the rich model, with `ll_arraycopy`
/// spelled as the item loop.
pub fn build_ll_listslice_startonly_graph(
    name: &str,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> FunctionGraph {
    let array_type_id = array_type_id.map(str::to_string);
    let mut graph = FunctionGraph::new(name);
    let start_block = graph.startblock;

    // Parameters: `l1`, `start`.
    let l1 = graph.alloc_value_var();
    let start = graph.alloc_value_var();
    for (var, param, ty) in [
        (&l1, "l1", ValueType::Ref(None)),
        (&start, "start", ValueType::Int),
    ] {
        graph.push_inputarg_var(start_block, var.clone());
        graph.push_op_with_result_var(
            start_block,
            OpKind::Input {
                name: param.to_string(),
                ty,
                class_root: None,
            },
            var.clone(),
        );
    }

    // len1 = l1.ll_length(); newlength = len1 - start;
    // l = RESLIST.ll_newlist(newlength)
    let len1 = push(
        &mut graph,
        start_block,
        OpKind::ArrayLen {
            base: l1.clone(),
            array_type_id: array_type_id.clone(),
            nolength: false,
        },
    );
    let newlength = push(
        &mut graph,
        start_block,
        OpKind::BinOp {
            op: "sub".into(),
            lhs: len1,
            rhs: start.clone(),
            result_ty: ValueType::Int,
        },
    );
    let new_list = push(
        &mut graph,
        start_block,
        OpKind::NewArrayClear {
            length: newlength.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
    );
    let zero = push(&mut graph, start_block, OpKind::ConstInt(0));

    // ll_arraycopy(l1, l, start, 0, newlength):
    //   i = 0
    //   while i < newlength: l[i] = l1[start + i]; i += 1
    let (head, head_args) = graph.create_block_with_arg_vars(5);
    let (body, body_args) = graph.create_block_with_arg_vars(5);
    let (done, done_args) = graph.create_block_with_arg_vars(1);
    graph.set_goto(
        start_block,
        head,
        vec![l1, start, new_list, newlength, zero],
    );

    let [_h_l1, _h_start, h_list, h_len, h_i] = head_args.as_slice() else {
        unreachable!("head block was created with five inputargs")
    };
    let in_range = push(
        &mut graph,
        head,
        OpKind::BinOp {
            op: "lt".into(),
            lhs: h_i.clone(),
            rhs: h_len.clone(),
            result_ty: ValueType::Bool,
        },
    );
    graph.set_branch(
        head,
        in_range,
        body,
        head_args.clone(),
        done,
        vec![h_list.clone()],
    );

    let [b_l1, b_start, b_list, b_len, b_i] = body_args.as_slice() else {
        unreachable!("body block was created with five inputargs")
    };
    let src_index = push(
        &mut graph,
        body,
        OpKind::BinOp {
            op: "add".into(),
            lhs: b_start.clone(),
            rhs: b_i.clone(),
            result_ty: ValueType::Int,
        },
    );
    let item = push(
        &mut graph,
        body,
        OpKind::ArrayRead {
            base: b_l1.clone(),
            index: src_index,
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: false,
            pure: false,
        },
    );
    graph.push_op_var(
        body,
        OpKind::ArrayWrite {
            base: b_list.clone(),
            index: b_i.clone(),
            value: LinkArg::Value(item),
            item_ty: item_ty.clone(),
            array_type_id,
            nolength: false,
        },
        false,
    );
    let one = push(&mut graph, body, OpKind::ConstInt(1));
    let next_i = push(
        &mut graph,
        body,
        OpKind::BinOp {
            op: "add".into(),
            lhs: b_i.clone(),
            rhs: one,
            result_ty: ValueType::Int,
        },
    );
    graph.set_goto(
        body,
        head,
        vec![
            b_l1.clone(),
            b_start.clone(),
            b_list.clone(),
            b_len.clone(),
            next_i,
        ],
    );

    // return l
    let [d_list] = done_args.as_slice() else {
        unreachable!("done block was created with one inputarg")
    };
    graph.set_return(done, Some(d_list.clone()));
    graph
}

fn push(graph: &mut FunctionGraph, block: crate::model::BlockId, kind: OpKind) -> Variable {
    graph
        .push_op_var(block, kind, true)
        .expect("a value-producing op allocates its result")
}

/// The op is the front's `__getslice_rangefrom(slice, start)` marker.
pub fn is_getslice_rangefrom(op: &SpaceOperation) -> bool {
    matches!(
        &op.kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments },
            args,
            ..
        } if segments.len() == 1 && segments[0] == "__getslice_rangefrom" && args.len() == 2
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CallTarget, ExitSwitch};

    #[test]
    fn helper_graph_has_the_rlist_shape() {
        let graph = build_ll_listslice_startonly_graph(
            "ll_listslice_startonly__ref",
            &ValueType::Ref(None),
            Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID),
        );
        // start / return / except / head / body / done.
        assert_eq!(graph.blocks.len(), 6);
        let start = graph.block(graph.startblock);
        assert_eq!(start.inputargs.len(), 2);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. }))
        );
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::NewArrayClear { .. }))
        );
        let head = graph.block(crate::model::BlockId(3));
        assert!(matches!(head.exitswitch, Some(ExitSwitch::Value(_))));
        assert_eq!(head.exits.len(), 2);
        let body = graph.block(crate::model::BlockId(4));
        assert!(
            body.operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayRead { .. }))
        );
        assert!(
            body.operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayWrite { .. }))
        );
        assert_eq!(body.exits.len(), 1);
        assert_eq!(body.exits[0].target, crate::model::BlockId(3));
        let done = graph.block(crate::model::BlockId(5));
        assert_eq!(done.exits[0].target, graph.returnblock);
    }

    #[test]
    fn array_identity_comes_from_a_sibling_read() {
        let mut graph = FunctionGraph::new("f");
        let args = graph.alloc_value_var();
        let index = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayRead {
                base: args.clone(),
                index,
                item_ty: ValueType::Ref(None),
                array_type_id: Some("objref".into()),
                nolength: false,
                pure: false,
            },
            true,
        );
        assert_eq!(
            array_identity_of_base(&graph, &args),
            Some((ValueType::Ref(None), Some("objref".into())))
        );
        let other = graph.alloc_value_var();
        assert_eq!(array_identity_of_base(&graph, &other), None);
    }

    #[test]
    fn array_identity_follows_the_value_across_links() {
        let mut graph = FunctionGraph::new("f");
        let args = graph.alloc_value_var();
        let index = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayRead {
                base: args.clone(),
                index,
                item_ty: ValueType::Ref(None),
                array_type_id: Some("objref".into()),
                nolength: false,
                pure: false,
            },
            true,
        );
        let (next, next_args) = graph.create_block_with_arg_vars(1);
        graph.set_goto(graph.startblock, next, vec![args]);
        assert_eq!(
            array_identity_of_base(&graph, &next_args[0]),
            Some((ValueType::Ref(None), Some("objref".into())))
        );
    }

    #[test]
    fn an_object_slice_input_names_its_own_identity() {
        let mut graph = FunctionGraph::new("f");
        let args = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, args.clone());
        graph.push_op_with_result_var(
            graph.startblock,
            OpKind::Input {
                name: "args_w".into(),
                ty: ValueType::Ref(None),
                class_root: Some(OBJECT_REF_SLICE_CLASS_ROOT.into()),
            },
            args.clone(),
        );
        assert_eq!(
            array_identity_of_base(&graph, &args),
            Some((ValueType::Ref(None), None))
        );
        let mut graph = FunctionGraph::new("g");
        let other = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, other.clone());
        graph.push_op_with_result_var(
            graph.startblock,
            OpKind::Input {
                name: "w_obj".into(),
                ty: ValueType::Ref(None),
                class_root: Some("PyObject".into()),
            },
            other.clone(),
        );
        assert_eq!(array_identity_of_base(&graph, &other), None);
    }

    #[test]
    fn a_bare_length_read_names_only_the_object_array() {
        let mut graph = FunctionGraph::new("f");
        let args = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayLen {
                base: args.clone(),
                array_type_id: Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID.into()),
                nolength: false,
            },
            true,
        );
        assert_eq!(
            array_identity_of_base(&graph, &args),
            Some((
                ValueType::Ref(None),
                Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID.into())
            ))
        );
        let mut graph = FunctionGraph::new("g");
        let bytes = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayLen {
                base: bytes.clone(),
                array_type_id: Some("bytes".into()),
                nolength: false,
            },
            true,
        );
        assert_eq!(array_identity_of_base(&graph, &bytes), None);
    }

    #[test]
    fn the_helper_is_minted_once_and_is_a_regular_callee() {
        use crate::codewriter::call::CallKind;
        let mut cc = CallControl::new();
        let first = listslice_startonly_path(&mut cc, &ValueType::Ref(None), Some("objref"));
        let again = listslice_startonly_path(&mut cc, &ValueType::Ref(None), Some("objref"));
        assert_eq!(first, again);
        let ints = listslice_startonly_path(&mut cc, &ValueType::Int, Some("ints"));
        assert_ne!(first, ints);
        let call = SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![first.last_segment().unwrap().to_string()],
                },
                args: vec![Variable::new(), Variable::new()],
                result_ty: ValueType::Ref(None),
            },
        };
        assert_eq!(cc.guess_call_kind(&call), CallKind::Regular);
    }
}
