//! Pre-jtransform fold of synthetic `__str_const` calls.
//!
//! `rtyper/rstr.py::StringRepr.convert_const` resolves a host string into a
//! prebuilt `Ptr(STR)` constant before `codewriter/jtransform` sees the graph.
//!
//! Pyre's frontend (`front::mir`) lowers string, char, and byte-string
//! literals to `OpKind::Call { target: FunctionPath(["__str_const",
//! payload]), args: [] }`. The companion fold inside
//! `translator/rtyper/flowspace_adapter.rs::legacy_const_define_hlvalue`
//! covers graphs that traverse the Match arm of the dual gate, but graphs
//! registered directly through `register_function_graph` can take the Skip
//! arm and bypass that fold.
//!
//! This pass operates directly on `model::FunctionGraph` before
//! `Transformer::transform`, so it catches both gate arms. The assembler
//! performs the lltype materialization, keeping low-level pointers out of
//! the model graph.
//!
//! # Deliberately dormant
//!
//! This pass is deliberately not wired into the translation pipeline yet.
//! Wiring it without the consumer half turns a string constant into a real
//! traced value that no string-taking consumer can read correctly; this has
//! been observed as a silent wrong answer in `pickle._dumps`. Before the pass
//! can be enabled, the JIT must have both a one-word string representation it
//! can carry and `jit_*` shims that let string-taking helpers be published or
//! looked inside. RPython strings are `Ptr(STR)`, a single word carrying its
//! own length, so upstream has no fat-pointer gap here.

use crate::model::{CallTarget, FunctionGraph, OpKind};

/// Rewrite zero-argument `__str_const` calls with valid WTF-8 payloads into
/// `OpKind::ConstStr`, mirroring `rtyper/rstr.py::StringRepr.convert_const`.
/// Invalid payloads remain calls because runtime string materialization uses
/// the same WTF-8 validity requirement.
///
/// The payload currently arrives as a JSON string through `Value::as_str` in
/// `front/mir.rs:18091`, so it is always valid UTF-8 and this guard cannot fire.
/// The guard stays so that a future byte-carrying representation leaves the
/// call symbolic and aborts the descent instead of reaching the runtime panic.
#[allow(dead_code)]
pub fn fold_str_consts(graph: &mut FunctionGraph) {
    for block in graph.blocks.iter_mut() {
        for op in block.operations.iter_mut() {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty()
                || segments.len() != 2
                || segments[0] != "__str_const"
                || rustpython_wtf8::Wtf8::from_bytes(segments[1].as_bytes()).is_none()
            {
                continue;
            }
            op.kind = OpKind::ConstStr(segments[1].as_bytes().to_vec());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ValueType;

    fn str_const_call(payload: String) -> OpKind {
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__str_const".to_string(), payload],
            },
            args: vec![],
            result_ty: ValueType::Ref(None),
        }
    }

    #[test]
    fn folds_valid_wtf8_and_preserves_result_variable() {
        let mut graph = FunctionGraph::new("valid_str_const");
        let entry = graph.startblock;
        let result = graph
            .push_op_var(entry, str_const_call("hello".to_string()), true)
            .expect("string constant call must produce a result");

        fold_str_consts(&mut graph);

        let op = &graph.block(entry).operations[0];
        assert_eq!(op.result.as_ref(), Some(&result));
        assert_eq!(op.kind, OpKind::ConstStr(b"hello".to_vec()));
    }
}
