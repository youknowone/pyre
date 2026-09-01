//! Every `unroll_safe` in the shipped interpreter LLBC is one that was
//! reviewed.
//!
//! `unroll_safe` is not a loop annotation.  `codewriter/policy.rs`
//! `look_inside_graph` cancels `contains_loop` for a hinted graph, so the
//! attribute changes *what the walker descends into*: the hinted graph and
//! its whole callee closure enter the candidate set.  The tree's history
//! with that is why this file exists — the same three attributes were added,
//! reverted for a SIGBUS traced to an sret ABI mismatch in a callee the hint
//! newly reached, and only re-landed once that callee was published
//! correctly.
//!
//! So an addition here is a descent-scope change that needs its own
//! evidence, and this test makes adding one without saying so fail.  It is
//! deliberately a *subset* check rather than an equality check: a developer's
//! `build/llbc` is routinely older than the source (a `pyre-interpreter`
//! edit is invisible until re-extraction), and a stale artefact must not
//! produce a false red.  A stale artefact can only under-report, which
//! passes; a new hint can only over-report, which fails.

use majit_charon_reader::Llbc;
use majit_translate::front::llbc_hints::harvest_hints_from_llbcs;

const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);

/// The leaf name of every function reviewed as `unroll_safe`, with the
/// upstream decorator it mirrors.
///
/// Matched on the leaf rather than the full path because the harvester
/// spells a method with its impl block (`pyframe::<Impl>::fast2locals`) and
/// a free function without one (`builtins::leading_non_null_count`); the
/// leaves are unambiguous across the interpreter.
const REVIEWED_UNROLL_SAFE: &[(&str, &str)] = &[
    // `abstractinst.py` carries `@jit.unroll_safe` on all three.
    (
        "isinstance",
        "abstractinst.py p_recursive_isinstance_w caller",
    ),
    ("issubclass", "abstractinst.py abstract_issubclass_w caller"),
    (
        "p_abstract_issubclass_w",
        "abstractinst.py _abstract_issubclass_w",
    ),
    // `pyframe.py` `fast2locals`.
    ("fast2locals", "pyframe.py fast2locals"),
    // `typeobject.py` `lookup_starting_at`, the MRO-suffix walk `super`
    // attribute lookup runs.  Without the hint `look_inside_graph` rejects the
    // graph for its loop and the whole lookup is one opaque residual.
    //
    // This is the descent-scope history in the file header repeating.  The hint
    // let the walk reach `wtf8_key_is_utf8`, published through
    // `jit_fnaddr.rs`'s ABI-unsound hatch because `&Wtf8` is two machine words
    // against the residual ABI's one, and executing it during the sub-walk
    // faulted.  Unlike the sret case, the callee cannot be published correctly
    // — the residual ABI has no fat-pointer argument at all — so the descent
    // declines at it instead (`is_abi_unsound_argument_residual`).
    //
    // Evidence for the descent-scope change: with that decline in place
    // `zero_arg_super_attr` matches its committed dynasm `.jitstats` baseline
    // on every counter, and the corpus gates unchanged against its baselines
    // with `fbw_rolled_back_with_effects` zero — a `JITSTATS_BADNESS_FIELDS`
    // member in `check.py`, gated corpus-wide, so a rise anywhere is a red.
    (
        "super_getattribute_wtf8",
        "typeobject.py lookup_starting_at",
    ),
    // No upstream counterpart by name; the loop is a bounded scan of a
    // fixed-size argument slice.
    ("leading_non_null_count", "flat builtin-keyword ABI scan"),
    // `argument.py` carries `@jit.unroll_safe` on `_match_signature`, the
    // keyword-binding loop this one mirrors; both are bounded by a signature
    // fixed at the callee rather than by the call's arguments.
    ("bind_builtin_kwargs", "argument.py _match_signature"),
    // `executioncontext.py` marks both `f_back`-chain walks `@jit.unroll_safe`.
    //
    // Evidence for the descent-scope change: neither graph is reachable from
    // the portal BFS today, so admitting them and their callee closures moved
    // no counter.  The whole corpus gates unchanged against its committed
    // `.jitstats` baselines on all three backends, and
    // `fbw_rolled_back_with_effects` stayed zero — it is a
    // `JITSTATS_BADNESS_FIELDS` member in `check.py`, so it is gated
    // corpus-wide rather than per fixture and a rise anywhere would have been
    // a red rather than a statistic.
    (
        "gettopframe_nohidden",
        "executioncontext.py gettopframe_nohidden",
    ),
    (
        "getnextframe_nohidden",
        "executioncontext.py getnextframe_nohidden",
    ),
    // No upstream counterpart: this materializes a 3.14 `FrameLocalsProxy`.
    // Its slot loop scans the same fixed-size locals-plus array `fast2locals`
    // does and takes the hint for the same reason — without it the function is
    // one residual call, and the `f_locals` read behind it forces the
    // virtualizable for that call's whole length.
    //
    // NOT the same evidence as the two above, and this line used to claim it
    // was: the body is REACHED.  `frame_locals_proxy_snapshot` holds a jitcode
    // in the metadata artefact, where `gettopframe_nohidden` and
    // `getnextframe_nohidden` do not.  The abort population and the baselines
    // were measured unchanged; unreachability is not why.
    (
        "frame_locals_proxy_snapshot",
        "fast2locals' scan of the same locals-plus array",
    ),
    // No upstream counterpart either: this is the mapping read behind a
    // `fr.f_locals["x"]`, and its scan is bounded by the same
    // `locals_plus_names` array — varnames plus cellvars plus freevars — that
    // `fast2locals` walks.  It cannot share `fast_local_index`, because a
    // running comprehension's iteration variable is readable through the proxy
    // even though assigning to that name goes to the extras dict.  What the
    // body does per candidate is not what the hint is about: `hash_w_strict`
    // and `eq_w` can run application code, but they are calls, not the loop
    // bound.
    //
    // Same class of evidence as `frame_locals_proxy_snapshot`, and for the
    // same reason it is NOT the `*_nohidden` argument: the body is REACHED.
    // `locals_plus_value` holds a jitcode in the generated metadata artefact,
    // so admitting it and its callee closure is a real descent-scope change
    // and unreachability proves nothing here.
    //
    // Measured in PR #1599's CI, on the tree that carries this attribute:
    // `pyre/check.py` passed in four jobs across all three runner OSes —
    // dynasm and cranelift on ubuntu-24.04, dynasm on windows-latest, and
    // dynasm+cranelift on macos-latest — with every committed per-fixture
    // jitstats baseline unchanged.  That covers `fbw_rolled_back_with_effects`
    // corpus-wide rather than per fixture, because it is a
    // `JITSTATS_BADNESS_FIELDS` member in `check.py`, so a rise anywhere would
    // have been a red rather than a statistic.
    (
        "locals_plus_value",
        "fast2locals' scan of the same locals-plus array, read through the proxy",
    ),
    // The undeduplicated `keys` / `values` / `items` scan behind the same
    // proxy, and the count `__len__` answers without materializing anything.
    // Both are bounded by the very `locals_plus_names` array `fast2locals`
    // walks; each pairs that green scan with an extras half that is bounded by
    // a dict length instead, which is why the extras half lives in its own
    // unhinted function (`pin_extra_locals_entries`) and stays a residual call.
    //
    // `entry_count` is named for what it counts rather than `length` because
    // this table matches by leaf and `length` is already the leaf of
    // `bufferview::BufferView::length`.
    //
    // Evidence for the descent-scope change: this is the `*_nohidden`
    // argument, not the `frame_locals_proxy_snapshot` one.  Neither graph
    // holds a jitcode in the generated metadata artefact -- `jit_metadata.json`
    // lists no `pin_entries`, and its one `length` jitcode resolves through
    // `symbolic_fnaddr_paths` to `bufferview::BufferView::length` -- so
    // admitting them and their callee closures reaches nothing the portal BFS
    // already walks.  Measured on the tree carrying both attributes:
    // `pyre/check.py` ran the full corpus on dynasm and cranelift, every
    // committed per-fixture jitstats baseline held except the seven this
    // Windows workstation also moves at `origin/main` with identical deltas,
    // and `fbw_rolled_back_with_effects` stayed zero -- it is a
    // `JITSTATS_BADNESS_FIELDS` member in `check.py`, so it is gated
    // corpus-wide rather than per fixture and a rise anywhere would have been
    // a red rather than a statistic.
    (
        "pin_entries",
        "fast2locals' scan of the same locals-plus array, undeduplicated",
    ),
    (
        "entry_count",
        "fast2locals' scan of the same locals-plus array, counted only",
    ),
    // `pyopcode.py` `dispatch_bytecode` is `@jit.unroll_safe`; its
    // EXTENDED_ARG loop is split here into the interpreter decoder and the
    // two scalar projections consumed by `eval_loop_jit`.  The projections
    // exist because the generated JIT's residual ABI cannot publish the
    // decoder's Result<(usize, Instruction, OpArg)> as one value.  Each loop
    // is bounded by the bytecode format's at-most-three EXTENDED_ARG units.
    //
    // Evidence for all three descent-scope changes: the complete cranelift
    // `pyre/check.py` corpus passed in PR #1580's CI with every committed
    // per-fixture jitstats baseline unchanged, including the corpus-wide
    // `fbw_rolled_back_with_effects == 0` gate.  The public decoder has no
    // jitcode in the generated metadata; both scalar projections do, so the
    // evidence covers the reached pair as well as the unreachable graph.
    (
        "decode_instruction_forward",
        "pyopcode.py dispatch_bytecode EXTENDED_ARG loop",
    ),
    (
        "decode_instruction_forward_pc",
        "pyopcode.py dispatch_bytecode EXTENDED_ARG loop (PC projection)",
    ),
    (
        "decode_instruction_forward_packed",
        "pyopcode.py dispatch_bytecode EXTENDED_ARG loop (opcode/oparg projection)",
    ),
    // `ctypefunc.py W_CTypeFunc._call` carries `@jit.unroll_safe`, and the two
    // halves it is split into here both take it: the body that converts the
    // arguments into the libffi exchange buffer, and the `finally` that
    // releases what those conversions allocated.  Both loops are bounded by
    // `len(self.fargs)`, which is a trace constant once the function type is
    // promoted off a constant cdata.
    //
    // Same class of evidence as `locals_plus_value`, and NOT the `*_nohidden`
    // argument: both bodies are REACHED — each holds a jitcode in the
    // generated metadata artefact — so unreachability proves nothing and the
    // corpus is what has to answer.  Measured on the tree that carries these
    // attributes, `pyre/check.py --backend dynasm` ran the whole synthetic and
    // macro corpus with every committed per-fixture jitstats baseline
    // unchanged.  That covers `fbw_rolled_back_with_effects` corpus-wide
    // rather than per fixture, because it is a `JITSTATS_BADNESS_FIELDS`
    // member in `check.py`, so a rise anywhere would have been a red rather
    // than a statistic.  The run's one failure was
    // `synth/load_name_builtin_cell_fold`'s pypy ratio against the
    // `max-pypy-ratio=1` that fixture carried at the time; `bench: lengthen
    // sub-50ms synthetic fixtures` raised it to 2.5 on main, and the measured
    // 1.6x passes the replacement.
    ("do_call", "ctypefunc.py W_CTypeFunc._call"),
    (
        "release_arguments",
        "ctypefunc.py W_CTypeFunc._call's finally block",
    ),
];

/// `builtins::leading_non_null_count` has carried its own `unroll_safe`
/// since it was introduced and appears in every recorded cache snapshot, so
/// its absence means the artefact is too old to say anything — skip loudly
/// rather than pass on a read that proves nothing.
const CONTROL: &str = "leading_non_null_count";

fn harvested_unroll_safe() -> Option<Vec<String>> {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file() {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-interpreter`"
        );
        return None;
    }
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let hints = harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
    let mut paths: Vec<String> = hints
        .iter()
        .filter(|(_, values)| values.iter().any(|h| h == "unroll_safe"))
        .map(|(path, _)| path.clone())
        .collect();
    paths.sort();
    if !paths.iter().any(|p| leaf(p) == CONTROL) {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} carries no `unroll_safe` on {CONTROL}, \
             so it predates the hint inventory entirely; re-extract to exercise \
             this test (harvested: {paths:?})"
        );
        return None;
    }

    // `REVIEWED_UNROLL_SAFE` and the subset check below both match on the
    // leaf, on the stated assumption that leaves are unambiguous across the
    // interpreter.  Nothing else verifies that.  If an unreviewed function
    // elsewhere later takes an already-reviewed leaf name and is hinted, the
    // subset check would pass on the strength of the other function's review
    // — the one outcome this file exists to prevent.
    let mut leaves: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for path in &paths {
        if let Some(previous) = leaves.insert(leaf(path), path.as_str()) {
            panic!(
                "leaf `{}` is ambiguous between {previous} and {path}. \
                 REVIEWED_UNROLL_SAFE matches by leaf, so it cannot tell them \
                 apart and one would ride on the other's review; key the \
                 inventory by full path before adding either.",
                leaf(path),
            );
        }
    }
    Some(paths)
}

fn leaf(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

#[test]
fn every_unroll_safe_in_the_shipped_llbc_is_a_reviewed_one() {
    let Some(paths) = harvested_unroll_safe() else {
        return;
    };
    for path in &paths {
        assert!(
            REVIEWED_UNROLL_SAFE
                .iter()
                .any(|(name, _)| *name == leaf(path)),
            "{path} carries `unroll_safe` but is not in REVIEWED_UNROLL_SAFE. \
             The attribute admits this graph and its callee closure into the \
             candidate set, so it needs its own evidence — measure \
             `fbw_rolled_back_with_effects` (a rise is a correctness verdict, \
             not a statistic) and the per-fixture jitstats before adding it, \
             then list it here. Harvested: {paths:?}",
        );
    }
}

/// `_unpackiterable_known_length_jitlook` quotes upstream's
/// `@jit.unroll_safe` in its own doc comment, which reads as an unfinished
/// port and has been picked up as one.  It is not.
///
/// Upstream hints that body for `unpackiterable_unroll`, whose
/// `expected_length` is an UNPACK_SEQUENCE oparg; `unpackiterable` reaches
/// it through `_unpackiterable_known_length`, which is
/// `@jit.dont_look_inside` — "the JIT stopped looking inside already".  pyre
/// has neither `unpackiterable_unroll` nor the shim, so `unpackiterable` is
/// the body's only caller: the hinted path upstream keeps closed.  Carrying
/// the attribute alone inverts that decision instead of matching it.
#[test]
fn the_known_length_unpack_body_stays_unhinted_without_its_shim() {
    let Some(paths) = harvested_unroll_safe() else {
        return;
    };
    assert!(
        !paths
            .iter()
            .any(|p| leaf(p) == "_unpackiterable_known_length_jitlook"),
        "`unroll_safe` on _unpackiterable_known_length_jitlook opens the path \
         upstream fences with the `@jit.dont_look_inside` shim \
         `_unpackiterable_known_length`, which pyre does not have. Port the \
         shim (and `unpackiterable_unroll`) with an ABI-correct publication \
         first — the signature returns `Result<Vec<PyObjectRef>, PyError>`, \
         which `helper_call_kind_for_type` answers `Unsupported` for. \
         Harvested: {paths:?}",
    );
}
