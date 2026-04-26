//! Port subset of `rpython/translator/backendopt/removenoops.py`.

use std::collections::HashMap;

use crate::flowspace::model::{FunctionGraph, Hlvalue};
use crate::translator::simplify;
use crate::translator::translator::TranslationContext;

/// RPython `remove_unaryops(graph, opnames)` at `removenoops.py:6-45`.
fn remove_unaryops(graph: &FunctionGraph, opnames: &[&str]) {
    let mut positions = Vec::new();
    for block in graph.iterblocks() {
        let b = block.borrow();
        for (i, op) in b.operations.iter().enumerate() {
            if opnames.contains(&op.opname.as_str()) {
                positions.push((block.clone(), i));
            }
        }
    }

    while let Some((block, index)) = positions.pop() {
        let (op_result, op_arg) = {
            let b = block.borrow();
            let op = &b.operations[index];
            (op.result.clone(), op.args[0].clone())
        };

        {
            let mut b = block.borrow_mut();
            for op in &mut b.operations[index..] {
                for arg in &mut op.args {
                    if *arg == op_result {
                        *arg = op_arg.clone();
                    }
                }
                if op.opname == "indirect_call"
                    && matches!(op.args.first(), Some(Hlvalue::Constant(_)))
                {
                    op.opname = "direct_call".to_string();
                    op.args.pop();
                }
            }
            for link in &b.exits {
                let mut link = link.borrow_mut();
                for arg in &mut link.args {
                    if arg.as_ref() == Some(&op_result) {
                        *arg = Some(op_arg.clone());
                    }
                }
            }
            if b.exitswitch.as_ref() == Some(&op_result) {
                match &op_arg {
                    Hlvalue::Variable(_) => b.exitswitch = Some(op_arg.clone()),
                    Hlvalue::Constant(c) => {
                        drop(b);
                        simplify::replace_exitswitch_by_constant(&block, c);
                        b = block.borrow_mut();
                    }
                }
            }
            b.operations.remove(index);
        }
    }
}

/// RPython `remove_same_as(graph)` at `removenoops.py:47-48`.
pub fn remove_same_as(graph: &FunctionGraph) {
    remove_unaryops(graph, &["same_as"]);
}

/// RPython `remove_duplicate_casts(graph, translator)` at
/// `removenoops.py:50-101`.
pub fn remove_duplicate_casts(graph: &FunctionGraph, translator: &TranslationContext) -> usize {
    simplify::join_blocks(graph);
    let mut num_removed = 0;

    for block in graph.iterblocks() {
        let mut comes_from: HashMap<Hlvalue, Hlvalue> = HashMap::new();
        let mut b = block.borrow_mut();
        for op in &mut b.operations {
            if op.opname == "cast_pointer" {
                if let Some(from_var) = comes_from.get(&op.args[0]).cloned() {
                    comes_from.insert(op.result.clone(), from_var.clone());
                    let same_type = concretetype_of(&from_var) == concretetype_of(&op.result);
                    if same_type {
                        op.opname = "same_as".to_string();
                        op.args = vec![from_var];
                        num_removed += 1;
                    } else {
                        op.args = vec![from_var];
                    }
                } else {
                    comes_from.insert(op.result.clone(), op.args[0].clone());
                }
            }
        }
    }
    if num_removed != 0 {
        remove_same_as(graph);
    }

    for block in graph.iterblocks() {
        let mut available: HashMap<(Hlvalue, Option<_>), Hlvalue> = HashMap::new();
        let mut b = block.borrow_mut();
        for op in &mut b.operations {
            if op.opname == "cast_pointer" {
                let key = (op.args[0].clone(), concretetype_of(&op.result));
                if let Some(existing) = available.get(&key).cloned() {
                    op.opname = "same_as".to_string();
                    op.args = vec![existing];
                    num_removed += 1;
                } else {
                    available.insert(key, op.result.clone());
                }
            }
        }
    }
    if num_removed != 0 {
        remove_same_as(graph);
        for block in graph.iterblocks() {
            let mut used: HashMap<Hlvalue, bool> = HashMap::new();
            {
                let b = block.borrow();
                for link in &b.exits {
                    for arg in &link.borrow().args {
                        if let Some(arg) = arg {
                            used.insert(arg.clone(), true);
                        }
                    }
                }
            }
            let mut b = block.borrow_mut();
            let mut i = b.operations.len();
            while i > 0 {
                i -= 1;
                let op = &b.operations[i];
                if op.opname == "cast_pointer" && !used.contains_key(&op.result) {
                    b.operations.remove(i);
                    num_removed += 1;
                } else {
                    let args = b.operations[i].args.clone();
                    for arg in args {
                        used.insert(arg, true);
                    }
                }
            }
        }
        let _ = translator.config.translation.verbose;
    }
    num_removed
}

/// RPython `remove_debug_assert(graph)` at `removenoops.py:103-107`.
pub fn remove_debug_assert(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        block
            .borrow_mut()
            .operations
            .retain(|op| op.opname != "debug_assert");
    }
}

fn concretetype_of(value: &Hlvalue) -> Option<crate::flowspace::model::ConcretetypePlaceholder> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}
