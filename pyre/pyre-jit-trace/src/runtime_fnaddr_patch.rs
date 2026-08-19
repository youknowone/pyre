//! Runtime patcher for stale build-time fnaddrs in deserialized JitCodes.
//!
//! RPython's translator AOT-compiles every helper into the same C binary as
//! the runtime metainterp, so `JitCode.fnaddr` and the funcptr entries the
//! codewriter materializes into `JitCode.constants_i` (`jtransform.py:455-471`
//! `handle_residual_call` + `:614-623 direct_funcptr_value`) are linker-
//! resolved C addresses that the runtime executes via `cpu.bh_call_*`
//! without further bookkeeping.
//!
//! Pyre's `majit-translate` runs in `pyre-jit-trace/build.rs` — a separate
//! cargo build-script process from `pyre-dynasm` (and any other pyre runtime
//! binary).  The fnaddrs the codewriter captured therefore reflect the
//! build-script process's `pyre_interpreter::jit_trace_fnaddrs()` snapshot,
//! whose addresses are invalidated by ASLR (per-process random slide) and by
//! the divergent executable layouts (the build-script binary embeds a
//! subset of the runtime's symbols).  A walker that follows
//! `execute_residual_call`'s elidable-EI branch
//! (`jitcode_dispatch::residual_call`'s `try_fold_pure_call_via_executor`)
//! into one of those stale addresses dereferences arbitrary memory →
//! SEGV.
//!
//! This module bridges that gap.  At build time
//! `pyre-jit-trace/build.rs` serialises the `(path, build_fnaddr)` table
//! that `pyre_interpreter::jit_trace_fnaddrs()` returned for the
//! codewriter.  At runtime [`patch_constants_i_fnaddrs`] re-queries
//! `jit_trace_fnaddrs()` (now reading the runtime process's addresses),
//! builds a `build_fnaddr → runtime_fnaddr` correspondence keyed by the
//! shared `path`, and rewrites every stale value in
//! `JitCode.constants_i` and `JitCode.fnaddr`.  After the patch the
//! walker's `call_int_function(funcptr, args)` invokes the correct
//! runtime entry point, matching the upstream linker-resolved
//! invariant.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use majit_translate::jitcode::JitCode;

/// Build-time `(path, build_fnaddr)` snapshot — bincoded by
/// `pyre-jit-trace/build.rs` from
/// `pyre_interpreter::jit_trace_fnaddrs()` immediately before the
/// codewriter consumes it.  Each entry shares its `path` with the
/// runtime call to `jit_trace_fnaddrs()` below; only the `i64` address
/// differs across processes.
fn build_time_fnaddr_bindings() -> Vec<(String, i64)> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fnaddr_bindings.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize fnaddr_bindings.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Apply the build → runtime fnaddr correspondence to every JitCode the
/// caller just deserialised.  Mutates each Arc in place — refcount must
/// be 1 on entry (the per-index loader satisfies this because
/// `bincode::deserialize` produces a fresh `Arc::new(...)` shell before any
/// consumer can clone it).
///
/// `JitCode.fnaddr` carries the shell-level fnaddr the codewriter
/// recorded in `CallControl::get_jitcode` (`codewriter/call.rs`); the per-
/// instruction funcptr operands the residual_call dispatcher reads
/// land in `JitCodeBody.constants_i` via the assembler's
/// `emit_const_i_from_const` path (`codewriter/assembler.rs`).  Both
/// surfaces are patched so the walker sees the same address regardless
/// of which lookup it routes through.
pub fn patch_constants_i_fnaddrs(jitcodes: &mut [Arc<JitCode>]) {
    let correspondence = &*FNADDR_CORRESPONDENCE;

    if correspondence.is_empty() {
        return;
    }

    for arc in jitcodes.iter_mut() {
        let jc = Arc::get_mut(arc).expect(
            "patch_constants_i_fnaddrs: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        if let Some(&runtime) = correspondence.get(&jc.fnaddr) {
            jc.fnaddr = runtime;
        }
        // Some shells reach the persisted table without a committed body
        // (e.g. `Default::default()` placeholders kept for `Arc<JitCode>::
        // default()` consumers in `BlackholeInterpreter::new`); they carry
        // empty `constants_i` so skipping the body-mut access is safe.
        if jc.try_body().is_some() {
            for c in jc.body_mut().constants_i.iter_mut() {
                if let Some(&runtime) = correspondence.get(c) {
                    *c = runtime;
                }
            }
        }
    }
}

static FNADDR_CORRESPONDENCE: LazyLock<HashMap<i64, i64>> = LazyLock::new(|| {
    let build_bindings = build_time_fnaddr_bindings();
    let runtime_bindings = pyre_interpreter::jit_trace_fnaddrs();

    let runtime_map: HashMap<&'static str, i64> = runtime_bindings.into_iter().collect();

    // `correspondence[build_fnaddr] = runtime_fnaddr` — only entries
    // whose runtime lookup actually disagrees with the build value get
    // patched; identical entries are dropped so the constants_i scan
    // can early-exit on a `HashMap::get` miss without comparing.
    //
    // The key is an address from the build process and the value one from
    // this one, so a build address can stand for two registered paths.  Both
    // ways that happens are benign, and last write wins in each:
    //
    //   * Several spellings of one function are registered on purpose
    //     (`jit_fnaddr.rs` lists both
    //     `pyre_object::listobject::jit_list_reverse` and
    //     `pyre_object::jit_list_reverse`); they resolve to the same runtime
    //     address, so the repeated insert is idempotent.
    //   * The build binary's identical-COMDAT folding (MSVC `/OPT:ICF`, or
    //     LLVM `MergeFunctions`) merges two byte-identical bodies onto one
    //     address.  With `#[inline(never)]` confined to the release-GIL
    //     surface the tracing-policy helpers inline their callees, and several
    //     distinct registered residuals then compile to the same code:
    //     `label_arg_to_usize` / `load_fast_var_num_to_index` are both
    //     `arg.get(op_arg).as_usize()`, `convert_value_arg` /
    //     `special_method_arg` are both `arg.get(op_arg)`, and
    //     `hash_str_hooked_bytes` decomposes its slice to the `(ptr, len)`
    //     `hash_str_hooked` already takes.  The build binary folds each pair;
    //     the runtime binary, built at a different optimization level, need
    //     not, so the two runtime addresses differ.  Folding merges only
    //     identical machine code, so a residual call patched to either twin
    //     runs the same body.
    //
    // A genuinely wrong runtime target could only come from a path bound to
    // the wrong `fn`, and that binding runs identically in both processes, so
    // the build and runtime addresses would agree rather than collide.  The
    // `jit_fnaddr` registry test guards against new same-address pairs but
    // observes only this process, where nothing folds, so it cannot see a
    // build-script fold.  There is therefore no collision this map must
    // reject; picking any of the folded twins' runtime addresses is correct.
    let mut correspondence: HashMap<i64, i64> = HashMap::new();
    for (path, build_fnaddr) in &build_bindings {
        let Some(&runtime_fnaddr) = runtime_map.get(path.as_str()) else {
            continue;
        };
        if *build_fnaddr != runtime_fnaddr {
            correspondence.insert(*build_fnaddr, runtime_fnaddr);
        }
    }

    correspondence
});

/// Build-time `(name, build_addr)` snapshot for the host `PyType` singleton
/// pointers the codewriter baked into `constants_i` (supplied through
/// `HostStaticAddrs.pytypes`). Same ASLR hazard + bincode round-trip as
/// [`build_time_fnaddr_bindings`].
fn build_time_pytype_bindings() -> Vec<(String, i64)> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/static_pytype_bindings.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize static_pytype_bindings.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Build-time `(name, build_addr)` snapshot for the prebuilt ref singletons
/// (`HostStaticAddrs.refs`).
fn build_time_ref_bindings() -> Vec<(String, i64)> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/static_ref_bindings.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize static_ref_bindings.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Rewrite stale build-time host-static addresses (`PyType` singletons and
/// prebuilt refs) the codewriter baked into the constant pools.  Mirrors
/// [`patch_constants_i_fnaddrs`] for the `HostStaticAddrs` *data* the body
/// references directly — e.g. `is_int`'s `ptr::eq((*obj).ob_type, &INT_TYPE)`
/// inlined into the `w_list_append` body, whose `&INT_TYPE` const was captured
/// in the build-script process and ASLR-invalidated at runtime.  Re-pairs each
/// build address with the runtime address from `jit_static_pytype_addrs` /
/// `jit_static_ref_addrs`, keyed by the shared name.  Both pools are scanned:
/// a host static used as a pointer-`eq` operand materializes as a `GcRef`
/// constant in `constants_r`, while one consumed as an integer lands in
/// `constants_i`.  `JitCode.fnaddr` is left untouched (these are data, not
/// call targets).
pub fn patch_static_addr_constants(jitcodes: &mut [Arc<JitCode>]) {
    let correspondence = &*STATIC_ADDR_CORRESPONDENCE;

    if correspondence.is_empty() {
        return;
    }

    for arc in jitcodes.iter_mut() {
        let jc = Arc::get_mut(arc).expect(
            "patch_static_addr_constants: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        if jc.try_body().is_some() {
            let body = jc.body_mut();
            for c in body
                .constants_i
                .iter_mut()
                .chain(body.constants_r.iter_mut())
            {
                if let Some(&runtime) = correspondence.get(c) {
                    *c = runtime;
                }
            }
        }
    }
}

static STATIC_ADDR_CORRESPONDENCE: LazyLock<HashMap<i64, i64>> = LazyLock::new(|| {
    let mut runtime_map: HashMap<&'static str, i64> = HashMap::new();
    runtime_map.extend(pyre_interpreter::jit_static_pytype_addrs());
    runtime_map.extend(pyre_interpreter::jit_static_ref_addrs());

    let mut correspondence: HashMap<i64, i64> = HashMap::new();
    for (name, build_addr) in build_time_pytype_bindings()
        .into_iter()
        .chain(build_time_ref_bindings())
    {
        if let Some(&runtime_addr) = runtime_map.get(name.as_str()) {
            if build_addr != runtime_addr {
                correspondence.insert(build_addr, runtime_addr);
            }
        }
    }

    correspondence
});

/// Take both build -> runtime address snapshots now.
///
/// Both maps are `LazyLock`, so without this they would be built at whatever
/// moment the first jitcode happens to be decoded.  Once jitcode bodies decode
/// lazily that moment is mid-trace, long after `init_typeobjects` has retagged
/// the `PyType` singletons — and the map, frozen there, would then patch every
/// later-decoded body's `constants_r` with addresses that no longer name what
/// the runtime holds.  Those constants reach a compiled loop's gc_table, so the
/// collector drags a bad root out of it (`GC BUG: invalid type_id`, site
/// `minor_root_target`).
///
/// Calling this from `ensure_finish_setup` restores the snapshot instant the
/// whole-table loader had, when the first `all_jitcodes()` decoded everything
/// against one consistent view.
pub fn prime_address_correspondences() {
    LazyLock::force(&FNADDR_CORRESPONDENCE);
    LazyLock::force(&STATIC_ADDR_CORRESPONDENCE);
}

/// Resolve one build-process function address to this process's address.
/// Used by the lazy indirect-call-target index, which needs the shell's
/// `fnaddr` but deliberately does not deserialize the shell's JitCode body.
pub(crate) fn runtime_fnaddr(build_fnaddr: i64) -> i64 {
    FNADDR_CORRESPONDENCE
        .get(&build_fnaddr)
        .copied()
        .unwrap_or(build_fnaddr)
}

/// High 16 bits of a deferred prebuilt-string sentinel (see
/// [`majit_translate::assembler::STR_CONST_SENTINEL_BASE`]).  x86-64 user
/// addresses occupy `0..2^48`, so a real GCREF / host-static address always
/// has these bits clear, while every sentinel has them set to the base
/// pattern.
const SENTINEL_HIGH_MASK: u64 = 0xFFFF_0000_0000_0000;

/// Materialize one immortal runtime `W_UnicodeObject` for a prebuilt-string
/// constant, returning its address.  `box_str_constant` leaks (never freed,
/// outside the nursery) a `W_UnicodeObject` whose `value: *mut Wtf8Buf`
/// indirection at `UNICODE_VALUE_OFFSET` is exactly what the trace readers
/// follow: `bh_strlen` / `bh_strgetitem` (`pyre_cpu.rs`) and the compiled
/// `PyreStrDescr` fast path both dereference that pointer, so the block is
/// indistinguishable from a `bh_newstr` result.  It is the same builder
/// `pyre-jit`'s `flatten.rs` uses for runtime string literals, and interns
/// identical literals by content (the runtime analog of the assembler's
/// per-jitcode dedup).  `precomputed_hash` is unused at runtime —
/// `W_UnicodeObject` carries no hash slot, so `ll_strhash` recomputes it from
/// `value` on demand.
fn materialize_prebuilt_str(bytes: &[u8], _precomputed_hash: i64) -> i64 {
    let wtf8 = rustpython_wtf8::Wtf8::from_bytes(bytes)
        .expect("prebuilt STR constant bytes are not valid WTF-8");
    pyre_object::unicodeobject::box_str_constant(wtf8) as i64
}

/// Materialize every deferred prebuilt-string constant the codewriter
/// recorded (`JitCodeBody.str_consts`, [`patch_constants_i_fnaddrs`]'s
/// sibling).  The build-time translator could not allocate a runtime STR
/// block, so it pooled a non-canonical sentinel in the slot named by each
/// descriptor's `constants_r_index`; here we allocate the immortal block and
/// overwrite the sentinel with its live address. Runs before the per-index
/// `OnceCell` publishes the entry — refcount must be 1 (`Arc::get_mut`), so no
/// consumer can observe the sentinel as a forged GCREF.
///
/// Identical literals are interned by bytes across the whole table so one
/// immortal block (one identity) is shared, the runtime analog of the
/// assembler's per-jitcode dedup.  `interned` is only a local fast path over
/// that: [`materialize_prebuilt_str`] resolves through
/// `box_str_constant`'s process-wide `STRING_INTERN_TABLE`, so the one-block
/// identity holds across calls even though entries are materialized one
/// jitcode at a time.
pub fn materialize_str_consts(jitcodes: &mut [Arc<JitCode>]) {
    let mut interned: HashMap<Vec<u8>, i64> = HashMap::new();
    for arc in jitcodes.iter_mut() {
        // Body-less placeholder shells, and bodies with no deferred strings
        // (the common case — only cutover string literals record any), need
        // no work and must not trip `Arc::get_mut` for nothing.
        if arc.try_body().is_none_or(|b| b.str_consts.is_empty()) {
            continue;
        }
        let jc = Arc::get_mut(arc).expect(
            "materialize_str_consts: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        let body = jc.body_mut();
        for i in 0..body.str_consts.len() {
            let idx = body.str_consts[i].constants_r_index;
            let hash = body.str_consts[i].precomputed_hash;
            let addr = {
                let bytes = &body.str_consts[i].bytes;
                if let Some(&a) = interned.get(bytes) {
                    a
                } else {
                    let owned = bytes.clone();
                    let a = materialize_prebuilt_str(&owned, hash);
                    interned.insert(owned, a);
                    a
                }
            };
            // The slot must still hold its non-canonical sentinel — never a
            // real address (which has the high bits clear).
            assert_eq!(
                (body.constants_r[idx] as u64) & SENTINEL_HIGH_MASK,
                (majit_translate::assembler::STR_CONST_SENTINEL_BASE as u64) & SENTINEL_HIGH_MASK,
                "constants_r[{idx}] did not hold a prebuilt-string sentinel",
            );
            body.constants_r[idx] = addr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_translate::assembler::STR_CONST_SENTINEL_BASE;
    use majit_translate::jitcode::{JitCode, JitCodeBody, StrConstDescriptor};

    fn sentinel(ordinal: i64) -> i64 {
        STR_CONST_SENTINEL_BASE | ordinal
    }

    /// Build a fresh `Arc<JitCode>` whose body carries `descs` plus a
    /// `constants_r` pre-seeded with the matching sentinels, mirroring the
    /// assembler's emit (ordinal == position in `str_consts`).
    fn jitcode_with_str_consts(descs: Vec<StrConstDescriptor>) -> Arc<JitCode> {
        let len = descs
            .iter()
            .map(|d| d.constants_r_index + 1)
            .max()
            .unwrap_or(0);
        let mut constants_r = vec![0_i64; len];
        for (ordinal, d) in descs.iter().enumerate() {
            constants_r[d.constants_r_index] = sentinel(ordinal as i64);
        }
        let jc = JitCode::new("test");
        jc.set_body(JitCodeBody {
            str_consts: descs,
            constants_r,
            ..Default::default()
        });
        Arc::new(jc)
    }

    #[test]
    fn materialize_str_consts_overwrites_sentinel_with_str_object() {
        use majit_ir::GcRef;
        use majit_metainterp::cpu::Cpu;

        let descs = vec![StrConstDescriptor {
            constants_r_index: 0,
            bytes: b"hello".to_vec(),
            precomputed_hash: 0x1234_5678,
        }];
        let mut jcs = vec![jitcode_with_str_consts(descs)];
        materialize_str_consts(&mut jcs);

        let addr = jcs[0].body().constants_r[0];
        assert_ne!(addr, sentinel(0), "sentinel must be overwritten");
        assert_eq!(
            (addr as u64) & SENTINEL_HIGH_MASK,
            0,
            "a real W_UnicodeObject address must have the sentinel high bits clear",
        );
        // Validate against the exact readers a live trace uses — the
        // `W_UnicodeObject.value` indirection at `UNICODE_VALUE_OFFSET`.  This is the
        // test that would have caught the old low-level-block layout bug:
        // `bh_strlen` follows the value pointer, so a non-`W_UnicodeObject` block
        // would read garbage / fault here.
        let cpu = crate::pyre_cpu::PyreCpu::new();
        assert_eq!(cpu.bh_strlen(GcRef(addr as usize)), Some(5));
        let got: Vec<u8> = (0..5)
            .map(|i| cpu.bh_strgetitem(GcRef(addr as usize), i).unwrap() as u8)
            .collect();
        assert_eq!(got, b"hello");
    }

    #[test]
    fn materialize_str_consts_interns_identical_bytes_across_jitcodes() {
        let desc = || StrConstDescriptor {
            constants_r_index: 0,
            bytes: b"x".to_vec(),
            precomputed_hash: 7,
        };
        let mut jcs = vec![
            jitcode_with_str_consts(vec![desc()]),
            jitcode_with_str_consts(vec![desc()]),
        ];
        materialize_str_consts(&mut jcs);
        let a0 = jcs[0].body().constants_r[0];
        let a1 = jcs[1].body().constants_r[0];
        assert_eq!(
            a0, a1,
            "identical literals must share one immortal W_UnicodeObject",
        );
        assert_ne!(a0, sentinel(0));
    }

    /// The lazy per-index loader materializes one entry per call, so the
    /// call-local `interned` map cannot be what shares the block between two
    /// jitcodes — `box_str_constant`'s process-wide intern table is.  Pins
    /// that the one-block identity survives the split into separate calls.
    #[test]
    fn materialize_str_consts_interns_across_separate_calls() {
        std::thread::spawn(|| {
            let desc = || StrConstDescriptor {
                constants_r_index: 0,
                bytes: b"y".to_vec(),
                precomputed_hash: 9,
            };
            let mut first = vec![jitcode_with_str_consts(vec![desc()])];
            let mut second = vec![jitcode_with_str_consts(vec![desc()])];
            materialize_str_consts(&mut first);
            materialize_str_consts(&mut second);
            assert_eq!(
                first[0].body().constants_r[0],
                second[0].body().constants_r[0],
                "identical literals must share one immortal W_UnicodeObject \
                 even when materialized one jitcode at a time",
            );
        })
        .join()
        .unwrap();
    }

    #[test]
    fn materialize_str_consts_empty_string() {
        use majit_ir::GcRef;
        use majit_metainterp::cpu::Cpu;

        let descs = vec![StrConstDescriptor {
            constants_r_index: 0,
            bytes: Vec::new(),
            precomputed_hash: -1,
        }];
        let mut jcs = vec![jitcode_with_str_consts(descs)];
        materialize_str_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0];
        let cpu = crate::pyre_cpu::PyreCpu::new();
        assert_eq!(cpu.bh_strlen(GcRef(addr as usize)), Some(0));
    }
}
