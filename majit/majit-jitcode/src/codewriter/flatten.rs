//! The parts of `rpython/jit/codewriter/flatten.py` the runtime codewriter
//! reads: the label carrier the assembler's switch descrs keep until labels
//! are resolved, and `reorder_renaming_list`.

use serde::{Deserialize, Serialize};

/// A label in the flattened instruction stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Label(pub usize);

/// `flatten.py class TLabel`.
///
/// Rust encodes the definition-vs-target distinction in `FlatOp` variants
/// (`Label` vs jump targets), so the target wrapper shares the same numeric
/// label carrier.
pub type TLabel = Label;

/// `flatten.py` `def reorder_renaming_list(frm, to):`.
///
/// Line-by-line port. Given two equal-length sequences `frm[i] -> to[i]`,
/// return an ordered list of `(src, dst)` pairs so that each move runs
/// after every read of its `dst` register has happened. Cycles are
/// broken by a `(src, None)` save and `(None, dst)` load pair:
///
/// ```py
/// def reorder_renaming_list(frm, to):
///     result = []
///     pending_indices = range(len(to))
///     while pending_indices:
///         not_read = dict.fromkeys([frm[i] for i in pending_indices])
///         still_pending_indices = []
///         for i in pending_indices:
///             if to[i] not in not_read:
///                 result.append((frm[i], to[i]))
///             else:
///                 still_pending_indices.append(i)
///         if len(pending_indices) == len(still_pending_indices):
///             # no progress -- there is a cycle
///             assert None not in not_read
///             result.append((frm[pending_indices[0]], None))
///             frm[pending_indices[0]] = None
///             continue
///         pending_indices = still_pending_indices
///     return result
/// ```
///
/// Each `(src, dst)` entry maps to one `%s_copy src -> dst` operation
/// emitted by `insert_renamings`; `(src, None)` maps to `%s_push src`
/// and `(None, dst)` maps to `%s_pop -> dst` (flatten.py).
///
/// `T: Eq + Clone + Hash` so the algorithm works for any register
/// representation — RPython uses `Register` objects keyed by identity,
/// we'll typically instantiate with `Register`, `u16` color indices,
/// or a mixed color/constant enum.
pub fn reorder_renaming_list<T>(frm: &[T], to: &[T]) -> Vec<(Option<T>, Option<T>)>
where
    T: Eq + Clone + std::hash::Hash,
{
    // Mutable copy so the `frm[pending_indices[0]] = None` cycle-break
    // write has a home. In Rust we use `Option<T>` in the working
    // buffer; `None` is the "register already saved on the stack"
    // marker, matching RPython's `frm[...] = None`.
    let mut frm: Vec<Option<T>> = frm.iter().cloned().map(Some).collect();
    let to: Vec<T> = to.to_vec();
    assert_eq!(frm.len(), to.len(), "frm and to must have equal length");

    let mut result: Vec<(Option<T>, Option<T>)> = Vec::new();
    // `pending_indices = range(len(to))`.
    let mut pending_indices: Vec<usize> = (0..to.len()).collect();

    // `while pending_indices:`.
    while !pending_indices.is_empty() {
        // `not_read = dict.fromkeys([frm[i] for i in pending_indices])`.
        // RPython builds a dict keyed on `frm[i]`; `None` entries mean
        // "already saved via push", which `to[i] not in not_read` checks
        // against.
        let not_read: std::collections::HashSet<Option<T>> =
            pending_indices.iter().map(|&i| frm[i].clone()).collect();
        let mut still_pending_indices: Vec<usize> = Vec::new();
        // `for i in pending_indices:`.
        for &i in &pending_indices {
            // `if to[i] not in not_read`.
            if !not_read.contains(&Some(to[i].clone())) {
                // `result.append((frm[i], to[i]))`.
                result.push((frm[i].clone(), Some(to[i].clone())));
            } else {
                // `still_pending_indices.append(i)`.
                still_pending_indices.push(i);
            }
        }
        // `if len(pending_indices) == len(still_pending_indices):`.
        if pending_indices.len() == still_pending_indices.len() {
            // `assert None not in not_read`.
            debug_assert!(
                !not_read.contains(&None),
                "reorder_renaming_list: duplicate cycle break"
            );
            // `result.append((frm[pending_indices[0]], None))`.
            let head = pending_indices[0];
            result.push((frm[head].clone(), None));
            // `frm[pending_indices[0]] = None`.
            frm[head] = None;
            continue;
        }
        pending_indices = still_pending_indices;
    }

    // After the main loop finishes, every `(src, None)` push needs a
    // matching `(None, dst)` pop at the tail of its cycle. RPython's
    // loop emits the pop naturally when the cycle's final read slot
    // becomes safe — but because `frm[head] = None` is an in-place
    // rewrite, the next iteration sees `frm[...] = None` and emits
    // `(None, to[...])` directly as part of `(frm[i], to[i])` above.
    // No separate pop stage needed.
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // `rpython/jit/codewriter/test/test_flatten.py` `test_reorder_renaming_list`.
    #[test]
    fn reorder_renaming_list_empty() {
        let result: Vec<(Option<i32>, Option<i32>)> = reorder_renaming_list::<i32>(&[], &[]);
        assert_eq!(result, Vec::<(Option<i32>, Option<i32>)>::new());
    }

    #[test]
    fn reorder_renaming_list_all_independent() {
        // No overlap between frm and to → identity order.
        let result = reorder_renaming_list(&[1, 2, 3], &[4, 5, 6]);
        assert_eq!(
            result,
            vec![(Some(1), Some(4)), (Some(2), Some(5)), (Some(3), Some(6)),]
        );
    }

    #[test]
    fn reorder_renaming_list_chain() {
        // 4→1, 5→2, 1→3, 2→4. Safe order: do (1→3) and (2→4) first
        // (their destinations aren't read later), then (4→1) and
        // (5→2). RPython expected: [(1,3), (4,1), (2,4), (5,2)].
        let result = reorder_renaming_list(&[4, 5, 1, 2], &[1, 2, 3, 4]);
        assert_eq!(
            result,
            vec![
                (Some(1), Some(3)),
                (Some(4), Some(1)),
                (Some(2), Some(4)),
                (Some(5), Some(2)),
            ]
        );
    }

    #[test]
    fn reorder_renaming_list_swap_cycle() {
        // 1↔2 is a cycle of length 2. Save 1 with push, do 2→1,
        // then pop→2. RPython expected: [(1,None), (2,1), (None,2)].
        let result = reorder_renaming_list(&[1, 2], &[2, 1]);
        assert_eq!(
            result,
            vec![(Some(1), None), (Some(2), Some(1)), (None, Some(2))]
        );
    }

    #[test]
    fn reorder_renaming_list_long_chain_and_two_cycles() {
        // Chain + two independent cycles: (7→8) safe;
        // (4→1, 3→2, 1→3, 2→4) is a 4-cycle; (6→5, 5→6) is a 2-cycle.
        let result = reorder_renaming_list(&[4, 3, 6, 1, 2, 5, 7], &[1, 2, 5, 3, 4, 6, 8]);
        assert_eq!(
            result,
            vec![
                (Some(7), Some(8)),
                (Some(4), None),
                (Some(2), Some(4)),
                (Some(3), Some(2)),
                (Some(1), Some(3)),
                (None, Some(1)),
                (Some(6), None),
                (Some(5), Some(6)),
                (None, Some(5)),
            ]
        );
    }
}
