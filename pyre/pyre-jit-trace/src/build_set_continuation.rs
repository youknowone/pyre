//! Resume tail for a BUILD_SET element `__hash__` inlined under the opcode.
//!
//! `crate::ctor_continuation` and `crate::operator_continuation` are the same
//! device; read those module docs first.  The register discipline and the
//! "entered only by resume" contract are identical.
//!
//! `try_walker_inline_build_set_from_array` records each element's `__hash__`
//! in the trace, then normalizes the digest and calls
//! `jit_walker_set_add_hashed`.  A guard inside that body does not resume
//! into those later ops.  `BhFrame::call_result_reg` is `code[position - 1]`,
//! so `_setup_return_value_r` (`blackhole.py`) writes the raw hash into
//! BUILD_SET's result register and the caller continues at `call.next_pc`.
//! The current element is never inserted, and neither are the ones after it.
//!
//! Upstream has no such gap.  `builtin_set_add_items_impl` is an ordinary
//! graph: `capture_resumedata` (`pyjitpl.py`) hands its frame over with the
//! rest of the framestack, and the blackhole runs the normalize, the
//! `wrap_set_element_hash_error` rewrite, and the remaining inserts.
//! Pyre synthesises the set at the opcode, the same way
//! `try_walker_inline_type_call` synthesises `descr_call`, so the tail has
//! to be spelled out:
//!
//! ```text
//!   0:          inline_call_r_r <placeholder>, set, item, array, index -> hash
//!   resume_pc:  -live-          [r0 set, r1 item, r2 array, r3 index]
//!               residual_call_r_r  bh_build_set_after_inlined_hash(...) -> set
//!               -live-          [nothing live]
//!               ref_return      set
//! ```
//!
//! The pending hash register is absent from the resume section.
//! `get_list_of_active_boxes(in_a_call=True)` (`pyjitpl.py`) clears a
//! caller's pending call-result slot, and `_setup_return_value_r` fills it
//! from `code[position - 1]` before the residual runs.  The four live boxes
//! are the set built so far, the element whose body just returned, the
//! original element array, and the boxed index of the next element.  Elements
//! before that index were inserted by the compiled trace; this residual must
//! not hash them again.
//!
//! The level is one of the paused parents on the `__hash__` frame
//! (`InlineFrame::parents`, outermost-first), so the chain is
//! `caller -> tail -> __hash__`.  The `inline_call` is never decoded.
//!
//! CONVERGENCE PATH: the same as `ctor_continuation`'s.  Entering
//! `bh_build_set_from_array`'s interpreter body as an ordinary callee, with
//! `__hash__` inlined at the `try_hash_value` inside
//! `builtin_set_add_items_impl`, makes this tail that graph's own jitcode.
//!
//! The array word is a `GcTypedArray` length prefix, not a `PyObject` header,
//! so the residual copies the not-yet-added elements out before the first
//! allocating call and does not publish that word as a root.  The resume
//! frame holds it until the copy.

use crate::PyJitCode;
use majit_metainterp::jitcode::{JitCallArg, JitCodeBuilder};

/// The set built by the elements hashed before this one.
pub(crate) const SET_REG: u16 = 0;
/// The element whose `__hash__` just returned.
pub(crate) const ITEM_REG: u16 = 1;
/// The BUILD_SET element array (`bh_build_set_from_array`'s argument).
pub(crate) const ARRAY_REG: u16 = 2;
/// Boxed index of the first element this residual still has to hash.
pub(crate) const INDEX_REG: u16 = 3;
/// Where the inlined `__hash__` return lands (`_setup_return_value_r`).
pub(crate) const HASH_RESULT_REG: u16 = 4;

/// Callee operand of the never-decoded `inline_call`.
const PLACEHOLDER_CALLEE: u16 = 0;

/// Finish one BUILD_SET element after its inlined `__hash__`, then hash the
/// elements the compiled trace has not inserted yet.
///
/// `normalize_hash_digest` failures go through `wrap_set_element_hash_error`
/// before they are published, matching `builtin_set_add_items_impl`.  A
/// `SetUpdateError` from `w_set_add_hashed_checked` is mapped by
/// `map_set_update_error`.  The null return is the residual ABI's failure
/// word; `jit_publish_residual_error_ref` has already filled both exception
/// channels.
pub extern "C" fn bh_build_set_after_inlined_hash(
    set: pyre_object::PyObjectRef,
    item: pyre_object::PyObjectRef,
    array: pyre_object::PyObjectRef,
    index_box: pyre_object::PyObjectRef,
    hash_obj: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let array_ptr = array as *const pyre_object::object_array::GcTypedArray;
    let start = resume_index(index_box);
    let len = pyre_object::object_array::gcarray_len(array_ptr);
    let mut pending = Vec::new();
    if start < len {
        pending.reserve(len - start);
        for index in start..len {
            pending.push(pyre_object::object_array::getarrayitem_ref(
                array_ptr, index,
            ));
        }
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let mut rooted = Vec::with_capacity(3 + pending.len());
    rooted.push(set);
    rooted.push(item);
    rooted.push(hash_obj);
    rooted.extend(pending);
    let base = pyre_object::gc_roots::pin_roots(&rooted);
    let hash = match pyre_interpreter::builtins::normalize_hash_digest(
        pyre_object::gc_roots::shadow_stack_get(base + 2),
    ) {
        Ok(hash) => hash,
        Err(err) => {
            return pyre_interpreter::runtime_ops::jit_publish_residual_error_ref(
                pyre_interpreter::baseobjspace::wrap_set_element_hash_error(
                    pyre_object::gc_roots::shadow_stack_get(base + 1),
                    err,
                ),
            );
        }
    };
    if let Err(err) = unsafe {
        pyre_object::w_set_add_hashed_checked(
            pyre_object::gc_roots::shadow_stack_get(base),
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            hash,
        )
    } {
        return pyre_interpreter::runtime_ops::jit_publish_residual_error_ref(
            pyre_interpreter::baseobjspace::map_set_update_error(err),
        );
    }
    let rest = rooted.len() - 3;
    for offset in 0..rest {
        let slot = base + 3 + offset;
        let hashed = match pyre_interpreter::builtins::try_hash_value(
            pyre_object::gc_roots::shadow_stack_get(slot),
        ) {
            Ok(hash) => hash,
            Err(err) => {
                return pyre_interpreter::runtime_ops::jit_publish_residual_error_ref(
                    pyre_interpreter::baseobjspace::wrap_set_element_hash_error(
                        pyre_object::gc_roots::shadow_stack_get(slot),
                        err,
                    ),
                );
            }
        };
        if let Err(err) = unsafe {
            pyre_object::w_set_add_hashed_checked(
                pyre_object::gc_roots::shadow_stack_get(base),
                pyre_object::gc_roots::shadow_stack_get(slot),
                hashed,
            )
        } {
            return pyre_interpreter::runtime_ops::jit_publish_residual_error_ref(
                pyre_interpreter::baseobjspace::map_set_update_error(err),
            );
        }
    }
    pyre_object::gc_roots::shadow_stack_get(base)
}

/// `index_box` is `w_int_new` of the next element.  A non-int or a negative
/// index adds no further elements: the compiled trace's own cursor is the
/// only one this residual trusts.
fn resume_index(index_box: pyre_object::PyObjectRef) -> usize {
    if index_box.is_null()
        || unsafe { !pyre_object::is_int(index_box) || pyre_object::is_bool(index_box) }
    {
        return usize::MAX;
    }
    let index = unsafe { pyre_object::w_int_get_value(index_box) };
    if index < 0 {
        usize::MAX
    } else {
        index as usize
    }
}

thread_local! {
    static LEVEL: std::cell::OnceCell<Option<(i32, usize)>> =
        const { std::cell::OnceCell::new() };
}

fn level() -> Option<(i32, usize)> {
    LEVEL.with(|cell| *cell.get_or_init(build))
}

/// The jitcode index of the tail, or `None` when it could not be built.
pub(crate) fn jitcode_index() -> Option<i32> {
    level().map(|(index, _)| index)
}

/// The byte offset the tail's resume section must name.
pub(crate) fn resume_pc() -> Option<usize> {
    level().map(|(_, pc)| pc)
}

fn build() -> Option<(i32, usize)> {
    let mut builder = JitCodeBuilder::new();
    builder.set_name("build_set_hash_tail");

    // Never decoded; present so `_setup_return_value_r` can read
    // `code[position-1]` as the dest register. See the module doc.
    builder.inline_call_r_r(
        PLACEHOLDER_CALLEE,
        &[
            (SET_REG, SET_REG),
            (ITEM_REG, ITEM_REG),
            (ARRAY_REG, ARRAY_REG),
            (INDEX_REG, INDEX_REG),
        ],
        Some(HASH_RESULT_REG),
    );

    let resume_pc = builder.current_pos();
    // `-live-` operands are 2-byte offsets into the ONE shared
    // `MetaInterpStaticData.liveness_info` pool (`pyjitpl.py`), so the triple
    // is interned there rather than in a private assembler buffer.  Box order
    // matches this ref list: set, item, array, index.
    let live = crate::state::intern_liveness(
        &[],
        &[
            SET_REG as u8,
            ITEM_REG as u8,
            ARRAY_REG as u8,
            INDEX_REG as u8,
        ],
        &[],
    )?;
    let live_patch = builder.live_placeholder();
    builder.patch_live_offset(live_patch, live);

    let funcptr = bh_build_set_after_inlined_hash as *const () as i64;
    let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr {
        arg_classes: "rrrrr".to_string(),
        result_type: 'r',
        ..Default::default()
    };
    let args = [
        JitCallArg {
            kind: majit_metainterp::jitcode::JitArgKind::Ref,
            reg: SET_REG,
        },
        JitCallArg {
            kind: majit_metainterp::jitcode::JitArgKind::Ref,
            reg: ITEM_REG,
        },
        JitCallArg {
            kind: majit_metainterp::jitcode::JitArgKind::Ref,
            reg: ARRAY_REG,
        },
        JitCallArg {
            kind: majit_metainterp::jitcode::JitArgKind::Ref,
            reg: INDEX_REG,
        },
        JitCallArg {
            kind: majit_metainterp::jitcode::JitArgKind::Ref,
            reg: HASH_RESULT_REG,
        },
    ];
    builder.residual_call_ref_canonical_typed_args(funcptr, &args, calldescr, SET_REG);
    // The residual's exception guard.  A failure carries the exception, and
    // `ref_return` runs only on the trace that passed the guard.
    let after_call = builder.live_placeholder();
    let empty = crate::state::intern_liveness(&[], &[], &[])?;
    builder.patch_live_offset(after_call, empty);
    builder.ref_return(SET_REG);

    // `try_finish` carries the builder's own `startpoints`, which every emit
    // helper above has already populated through `start_instr`.  The set is
    // every instruction start, not just the addressable ones: `run_inner`
    // asserts membership on EVERY dispatched position under
    // `jit_strict_mode`, so narrowing it would panic at the first op after
    // the resume anchor.
    let jitcode = builder.try_finish()?;

    let payload = std::sync::Arc::new(PyJitCode::from_core_degenerate(
        std::sync::Arc::new(jitcode),
        std::ptr::null(),
        /* has_abort */ false,
    ));
    let index = crate::state::install_codeless_jitcode(payload);
    Some((index, resume_pc))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_exc() {
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|cell| cell.set(0));
    }

    #[test]
    fn resume_anchor_sits_behind_the_inline_call_return_tail() {
        let Some((index, resume_pc)) = level() else {
            return;
        };
        let payload = crate::state::pyjitcode_for_jitcode_index(index)
            .expect("the tail was just installed at this index");
        let code = payload.jitcode.code.as_slice();
        assert!(
            resume_pc >= 3,
            "no room for the call header and its return register"
        );
        assert_eq!(
            code[resume_pc - 1] as u16,
            HASH_RESULT_REG,
            "the last operand is the hash result register",
        );
        assert_eq!(
            code[resume_pc],
            crate::state::op_live(),
            "the resume position must be the `-live-` anchor itself",
        );
        let startpoints = payload
            .jitcode
            .startpoints
            .as_ref()
            .expect("the builder records one startpoint per emitted instruction");
        assert!(
            startpoints.contains(&resume_pc),
            "the anchor must be addressable as a resume coordinate",
        );
        // `inline_call`, `-live-`, `residual_call`, the guard's `-live-`,
        // `ref_return`.
        assert_eq!(
            startpoints.len(),
            5,
            "startpoints must cover every op in the tail, got {startpoints:?}",
        );
        assert!(
            payload
                .jitcode
                .can_decode_live_vars(resume_pc, crate::state::op_live()),
            "the anchor must decode its live vars",
        );
        assert_eq!(
            payload.resolve_resume_pc_with_jitcode_pc(resume_pc as i32, crate::state::op_live()),
            Some(resume_pc),
            "the parent loop must resolve the anchor to itself",
        );
    }

    #[test]
    fn adds_the_current_item_and_the_unhashed_tail() {
        clear_exc();
        let set = pyre_object::w_set_new();
        let item = pyre_object::w_int_new(7);
        let hash_obj = pyre_object::w_int_new(7);
        let array = pyre_object::object_array::allocate_array(
            1,
            pyre_object::object_array::ArrayKind::Ref,
            true,
        );
        pyre_object::object_array::setarrayitem_ref(array, 0, pyre_object::w_int_new(8));
        let result = bh_build_set_after_inlined_hash(
            set,
            item,
            array as pyre_object::PyObjectRef,
            pyre_object::w_int_new(0),
            hash_obj,
        );
        assert!(!result.is_null());
        assert_eq!(unsafe { pyre_object::w_set_len(result) }, 2);
        assert_eq!(
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|cell| cell.get()),
            0
        );
    }

    #[test]
    fn an_index_past_the_array_adds_only_the_current_item() {
        clear_exc();
        let set = pyre_object::w_set_new();
        let array = pyre_object::object_array::allocate_array(
            1,
            pyre_object::object_array::ArrayKind::Ref,
            true,
        );
        pyre_object::object_array::setarrayitem_ref(array, 0, pyre_object::w_int_new(8));
        let result = bh_build_set_after_inlined_hash(
            set,
            pyre_object::w_int_new(7),
            array as pyre_object::PyObjectRef,
            pyre_object::w_int_new(1),
            pyre_object::w_int_new(7),
        );
        assert!(!result.is_null());
        assert_eq!(unsafe { pyre_object::w_set_len(result) }, 1);
    }

    #[test]
    fn a_non_integer_digest_names_the_set_element() {
        // `str()` of the published exception reads subclass ranges and the
        // type registry. A filtered run of this module never reaches another
        // test's `init_typeobjects`, and with both unpublished `ll_isinstance`
        // declines the instance and `type()` is missing, so the diagnostic
        // repr is `<object object at ...>` even when `args_w` holds the wrap.
        pyre_interpreter::typedef::init_typeobjects();
        clear_exc();
        let array = pyre_object::object_array::allocate_array(
            0,
            pyre_object::object_array::ArrayKind::Ref,
            true,
        );
        let result = bh_build_set_after_inlined_hash(
            pyre_object::w_set_new(),
            pyre_object::w_int_new(7),
            array as pyre_object::PyObjectRef,
            pyre_object::w_int_new(0),
            pyre_object::w_str_new("nope"),
        );
        assert!(result.is_null());
        let exc = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|cell| cell.get())
            as pyre_object::PyObjectRef;
        assert!(!exc.is_null());
        // Nursery exception (`alloc_exception_nursery`). `message_text` allocates
        // the args tuple; pin first so that collection rewrites this slot.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = pyre_object::gc_roots::pin_roots(&[exc]);
        let err = unsafe {
            pyre_interpreter::PyError::from_exc_object(pyre_object::gc_roots::shadow_stack_get(
                slot,
            ))
        };
        let message = err.message_text();
        assert!(
            message.contains("cannot use 'int' as a set element"),
            "wrapped message was {message}",
        );
        assert!(
            message.contains("should return an integer"),
            "wrapped message was {message}",
        );
        clear_exc();
    }
}
