use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
/// runner.py: AbstractX86CPU — the Backend trait implementation.
///
/// This is the entry point for the dynasm backend, corresponding to
/// rpython/jit/backend/x86/runner.py AbstractX86CPU.
use std::sync::atomic::Ordering;

use majit_backend::{AsmInfo, Backend, BackendError, DeadFrame, ExitRecoveryLayout, JitCellToken};
use majit_ir::{FailDescr, GcRef, InputArg, Op, OpRef, Type, Value};

#[cfg(target_arch = "aarch64")]
use crate::aarch64::assembler::{AssemblerARM64 as Asm, CompiledCode};
use crate::arch;
use crate::codebuf;
use crate::frame::FrameData;
use crate::guard::DynasmFailDescr;
use crate::jitframe::JitFrame;
#[cfg(target_arch = "x86_64")]
use crate::x86::assembler::{Assembler386 as Asm, CompiledCode};

/// Global CALL_ASSEMBLER target registry.
///
/// RPython stores `descr._ll_function_addr` on the target token
/// (x86/assembler.py:599) so `CALL_ASSEMBLER` can resolve the callee
/// address directly from the descriptor. pyre identifies callee tokens
/// by `u64 token_number` inside `MetaCallAssemblerDescr` (pyre PRE-
/// EXISTING-ADAPTATION — serializable descriptors), so a process-wide
/// `token_number -> _ll_function_addr` index is required.
///
/// Metadata orthodox to `CompiledLoopToken` (`_ll_initial_locs`,
/// `frame_info`, `index_of_virtualizable`) lives on the token itself,
/// not in this registry — `handle_call_assembler` (rewrite.py:665-695)
/// reads it from `token.compiled_loop_token`.
static CALL_ASSEMBLER_TARGETS: LazyLock<Mutex<HashMap<u64, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

thread_local! {
    /// llmodel.py self.gc_ll_descr — owned by the active dynasm
    /// backend on this thread. Stored as a thread-local so the
    /// backend-agnostic `majit_gc::ActiveGcGuardHooks` shims can
    /// reach the live allocator without taking a dynasm dependency.
    pub static DYNASM_ACTIVE_GC: RefCell<Option<Box<dyn majit_gc::GcAllocator>>> =
        const { RefCell::new(None) };
}

fn with_dynasm_active_gc<R>(f: impl FnOnce(&dyn majit_gc::GcAllocator) -> R) -> Option<R> {
    DYNASM_ACTIVE_GC.with(|cell| {
        let guard = cell.borrow();
        guard.as_deref().map(f)
    })
}

fn dynasm_check_is_object(gcref: GcRef) -> bool {
    with_dynasm_active_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
}

fn dynasm_get_actual_typeid(gcref: GcRef) -> Option<u32> {
    with_dynasm_active_gc(|gc| gc.get_actual_typeid(gcref)).flatten()
}

fn dynasm_subclass_range(classptr: usize) -> Option<(i64, i64)> {
    with_dynasm_active_gc(|gc| gc.subclass_range(classptr)).flatten()
}

fn dynasm_typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    with_dynasm_active_gc(|gc| gc.typeid_subclass_range(typeid)).flatten()
}

/// gc.py:525-531 `get_nursery_free_addr` / `get_nursery_top_addr` parity:
/// the backend reads nursery slot addresses from the active GC descriptor,
/// NOT from a process-global singleton. Returns `(0, 0)` when no GC is
/// bound so the assembler falls back to the slow-path helper.
pub(crate) fn dynasm_nursery_addrs() -> (usize, usize) {
    with_dynasm_active_gc(|gc| (gc.nursery_free_addr(), gc.nursery_top_addr())).unwrap_or((0, 0))
}

/// Per-backend `CPU.load_supported_factors` (rewrite.py:1124 /
/// x86/runner.py:31 / llmodel.py:39). x86 addressing scales natively by
/// 1/2/4/8, aarch64 has no scaled store form and always expects factor 1.
#[cfg(target_arch = "x86_64")]
fn gc_store_supported_factors() -> &'static [i64] {
    &[1, 2, 4, 8]
}

#[cfg(target_arch = "aarch64")]
fn gc_store_supported_factors() -> &'static [i64] {
    &[1]
}

fn dynasm_typeid_is_object(typeid: u32) -> Option<bool> {
    with_dynasm_active_gc(|gc| gc.typeid_is_object(typeid)).flatten()
}

/// _build_malloc_slowpath parity: nursery overflow slow path.
///
/// Called from JIT-compiled code when inline nursery bump allocation
/// fails (new_free > nursery_top). total_size includes GcHeader.
///
/// Returns payload pointer (after GcHeader), matching fast-path semantics.
pub extern "C" fn dynasm_nursery_slowpath(total_size: u64) -> u64 {
    let gc_hdr = majit_gc::header::GcHeader::SIZE;
    let result = DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        guard
            .as_mut()
            .map(|gc| gc.alloc_nursery(total_size as usize - gc_hdr).0 as u64)
    });
    result.unwrap_or_else(|| unsafe {
        let raw = libc::calloc(1, total_size as usize) as u64;
        raw + gc_hdr as u64
    })
}

/// _build_malloc_slowpath(kind='var') parity: varsize nursery overflow.
/// Called with (base_size, item_size, length). Returns payload pointer.
pub extern "C" fn dynasm_nursery_slowpath_varsize(
    base_size: u64,
    item_size: u64,
    length: u64,
) -> u64 {
    let gc_hdr = majit_gc::header::GcHeader::SIZE;
    let result = DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        guard.as_mut().map(|gc| {
            gc.alloc_varsize(base_size as usize, item_size as usize, length as usize)
                .0 as u64
        })
    });
    result.unwrap_or_else(|| {
        let total = base_size as usize + item_size as usize * length as usize + gc_hdr;
        unsafe {
            let raw = libc::calloc(1, total) as u64;
            raw + gc_hdr as u64
        }
    })
}

/// opassembler.py:956-976: non-array write barrier slow path.
/// Calls gc.write_barrier(obj) which is the generic barrier.
pub extern "C" fn dynasm_write_barrier(obj_ptr: u64) {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_mut() {
            gc.write_barrier(majit_ir::GcRef(obj_ptr as usize));
        }
    });
}

/// opassembler.py:953-960: array write barrier slow path.
/// Calls jit_remember_young_pointer_from_array(obj) which handles
/// the CARDS_SET transition for HAS_CARDS arrays.
pub extern "C" fn dynasm_write_barrier_from_array(obj_ptr: u64) {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_mut() {
            gc.jit_remember_young_pointer_from_array(majit_ir::GcRef(obj_ptr as usize));
        }
    });
}

/// runner.py:23 AbstractX86CPU — concrete Backend implementation.
pub struct DynasmBackend {
    /// Next unique trace ID.
    next_trace_id: u64,
    /// Next header PC (green key).
    next_header_pc: u64,
    /// Constants for the next compilation.
    constants: std::collections::HashMap<u32, i64>,
    /// Constant type annotations for GC rewriter.
    constant_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// llmodel.py:64-69 self.vtable_offset — byte offset of the typeptr
    /// field inside instance objects. None when gcremovetypeptr is enabled.
    vtable_offset: Option<usize>,
}

impl DynasmBackend {
    #[inline]
    fn raw_mem_ptr(addr: i64, offset: i64) -> usize {
        assert_ne!(
            addr, 0,
            "llmodel.py parity: raw memory helpers must not silently accept NULL addresses"
        );
        (addr as usize).wrapping_add(offset as usize)
    }

    /// llmodel.py:467-478 read_int_at_mem(gcref, ofs, size, sign).
    fn read_int_at_mem(&self, addr: i64, offset: i64, size: usize, sign: bool) -> i64 {
        let ptr = Self::raw_mem_ptr(addr, offset);
        unsafe {
            match (size, sign) {
                (1, true) => (ptr as *const i8).read_unaligned() as i64,
                (1, false) => (ptr as *const u8).read_unaligned() as i64,
                (2, true) => (ptr as *const i16).read_unaligned() as i64,
                (2, false) => (ptr as *const u16).read_unaligned() as i64,
                (4, true) => (ptr as *const i32).read_unaligned() as i64,
                (4, false) => (ptr as *const u32).read_unaligned() as i64,
                _ => (ptr as *const i64).read_unaligned(),
            }
        }
    }

    /// llmodel.py:481-488 write_int_at_mem(gcref, ofs, size, newvalue).
    fn write_int_at_mem(&self, addr: i64, offset: i64, size: usize, newvalue: i64) {
        let ptr = Self::raw_mem_ptr(addr, offset);
        unsafe {
            match size {
                1 => (ptr as *mut u8).write_unaligned(newvalue as u8),
                2 => (ptr as *mut u16).write_unaligned(newvalue as u16),
                4 => (ptr as *mut u32).write_unaligned(newvalue as u32),
                _ => (ptr as *mut i64).write_unaligned(newvalue),
            }
        }
    }

    /// llmodel.py:490-491 read_float_at_mem(gcref, ofs).
    fn read_float_at_mem(&self, addr: i64, offset: i64) -> f64 {
        let ptr = Self::raw_mem_ptr(addr, offset);
        unsafe { (ptr as *const f64).read_unaligned() }
    }

    /// llmodel.py:493-494 write_float_at_mem(gcref, ofs, newvalue).
    fn write_float_at_mem(&self, addr: i64, offset: i64, newvalue: f64) {
        let ptr = Self::raw_mem_ptr(addr, offset);
        unsafe { (ptr as *mut f64).write_unaligned(newvalue) }
    }

    pub fn new() -> Self {
        DynasmBackend {
            next_trace_id: 1,
            next_header_pc: 0,
            constants: std::collections::HashMap::new(),
            constant_types: std::collections::HashMap::new(),
            vtable_offset: None,
        }
    }

    /// Active vtable_offset for the assembler to consume during codegen.
    pub fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    // `set_constants`, `set_constant_types`, `set_next_trace_id`,
    // `set_next_header_pc` are provided via the `Backend` trait impl
    // below so `compile_tmp_callback` and other backend-agnostic
    // consumers can reach them through `&mut dyn Backend`.

    /// Stub — dynasm doesn't need GC runtime ID.
    pub fn gc_runtime_id(&self) -> Option<u64> {
        None
    }
    pub fn set_gc_runtime_id(&mut self, _id: u64) {}

    /// gc.py:525-531 parity: build a GcRewriterImpl from the active GC.
    fn gc_rewriter(
        &self,
        constant_types: &std::collections::HashMap<u32, majit_ir::Type>,
    ) -> Option<majit_gc::rewrite::GcRewriterImpl> {
        with_dynasm_active_gc(|gc| {
            let ct = constant_types.clone();
            majit_gc::rewrite::GcRewriterImpl {
                nursery_free_addr: gc.nursery_free_addr(),
                nursery_top_addr: gc.nursery_top_addr(),
                max_nursery_size: gc.max_nursery_object_size(),
                wb_descr: {
                    let mut descr = majit_gc::WriteBarrierDescr::for_current_gc();
                    let card_page_shift = gc.card_page_shift();
                    if card_page_shift > 0 {
                        descr.jit_wb_card_page_shift = card_page_shift;
                    } else {
                        descr.jit_wb_cards_set = 0;
                        descr.jit_wb_card_page_shift = 0;
                        descr.jit_wb_cards_set_byteofs = 0;
                        descr.jit_wb_cards_set_singlebyte = 0;
                    }
                    descr
                },
                jitframe_info: None,
                constant_types: ct,
                call_assembler_callee_locs: None,
                // x86/runner.py:31 `load_supported_factors = (1, 2, 4, 8)`
                // vs llmodel.py:39 default `(1,)` used by the aarch64
                // backend (which has no scaled store addressing mode and
                // asserts boxes[3].getint() == 1 in its regalloc — see
                // `consider_gc_store_indexed` cfg(target_arch = "aarch64")).
                load_supported_factors: gc_store_supported_factors(),
            }
        })
    }

    /// rewrite.py:345 parity: run GC rewriter on ops before assembly.
    fn prepare_ops_for_compile(&mut self, inputargs: &[InputArg], ops: &[Op]) -> Vec<Op> {
        let num_inputs = inputargs.len() as u32;
        let mut normalized: Vec<Op> = ops
            .iter()
            .enumerate()
            .map(|(op_idx, op)| {
                let mut n = op.clone();
                if n.result_type() != Type::Void && n.pos.is_none() {
                    n.pos = OpRef(num_inputs + op_idx as u32);
                }
                n
            })
            .collect();
        // rewrite.py:489 parity: inject str_descr/unicode_descr for NEWSTR/NEWUNICODE
        inject_builtin_string_descrs(&mut normalized);
        let constant_types = std::mem::take(&mut self.constant_types);
        if let Some(rewriter) = self.gc_rewriter(&constant_types) {
            use majit_gc::GcRewriter;
            let constants = &self.constants;
            let (result, new_constants) =
                rewriter.rewrite_for_gc_with_constants(&normalized, constants);
            for (k, v) in new_constants {
                self.constants.entry(k).or_insert(v);
            }
            result
        } else {
            normalized
        }
    }

    /// llmodel.py:53-54: store gc_ll_descr on the cpu instance.
    ///
    /// Dynasm does not have cranelift's runtime-id indirection, so it
    /// mirrors wasm: the live allocator is stored in a thread-local and
    /// exposed through backend-agnostic `majit_gc::ActiveGcGuardHooks`.
    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn majit_gc::GcAllocator>) {
        gc.freeze_types();
        let supports_guard_gc_type = gc.supports_guard_gc_type();
        DYNASM_ACTIVE_GC.with(|cell| *cell.borrow_mut() = Some(gc));
        majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
            check_is_object: Some(dynasm_check_is_object),
            get_actual_typeid: Some(dynasm_get_actual_typeid),
            subclass_range: Some(dynasm_subclass_range),
            typeid_subclass_range: Some(dynasm_typeid_subclass_range),
            typeid_is_object: Some(dynasm_typeid_is_object),
            supports_guard_gc_type,
        });
    }

    /// llmodel.py:64-69 self.vtable_offset configuration.
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.vtable_offset = offset;
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer to its registered GC type id via the
    /// installed gc_ll_descr (the GC backend supplied through
    /// set_gc_allocator).
    pub fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        with_dynasm_active_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr))
            .flatten()
    }

    /// Pre-compute classptr → expected_typeid pairs for every GuardClass /
    /// GuardNonnullClass operand seen in `ops`. RPython resolves these on
    /// demand inside `_cmp_guard_class` (assembler.py:1887-1890); pyre's
    /// dynasm assembler runs without a borrow of `self`, so we materialize
    /// the resolver as a HashMap up front.
    fn collect_classptr_typeid_table(
        &self,
        ops: &[Op],
        constants: &std::collections::HashMap<u32, i64>,
    ) -> std::collections::HashMap<i64, u32> {
        let mut table = std::collections::HashMap::new();
        if self.vtable_offset.is_some() || DYNASM_ACTIVE_GC.with(|cell| cell.borrow().is_none()) {
            // vtable_offset path doesn't need typeid lookups; without a
            // gc_ll_descr there is nothing to resolve anyway.
            return table;
        }
        for op in ops {
            if matches!(
                op.opcode,
                majit_ir::OpCode::GuardClass | majit_ir::OpCode::GuardNonnullClass
            ) && op.args.len() >= 2
            {
                if let Some(&classptr) = constants.get(&op.args[1].0) {
                    if let Some(tid) = self.lookup_typeid_from_classptr(classptr as usize) {
                        table.insert(classptr, tid);
                    }
                }
            }
        }
        table
    }

    fn get_compiled(token: &JitCellToken) -> &CompiledCode {
        token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledCode>()
            .expect("compiled data is not CompiledCode")
    }

    fn input_slot(position: usize) -> usize {
        arch::JITFRAME_FIXED_SIZE + position
    }

    /// llmodel.py:412 get_latest_descr parity: resolve a raw jf_descr
    /// pointer to its Arc<DynasmFailDescr>. Searches root loop
    /// fail_descrs first, then all bridge fail_descrs stored in
    /// asmmemmgr_blocks. RPython does this via AbstractDescr.show()
    /// which works for any descr from any loop/bridge.
    ///
    /// Panics if not found — RPython uses object identity, so lookup
    /// failure is impossible in well-formed execution.
    fn find_descr_by_ptr(token: &JitCellToken, ptr: usize) -> Arc<DynasmFailDescr> {
        // compile.py:618-669 done_with_this_frame_descr — check all 4 variants
        if crate::guard::is_done_with_this_frame_descr(ptr) {
            // Determine type from which variant matched
            let types = if ptr == crate::guard::done_with_this_frame_descr_void_ptr() {
                vec![]
            } else if ptr == crate::guard::done_with_this_frame_descr_float_ptr() {
                vec![Type::Float]
            } else if ptr == crate::guard::done_with_this_frame_descr_ref_ptr() {
                vec![Type::Ref]
            } else {
                vec![Type::Int]
            };
            return Arc::new(DynasmFailDescr::new(u32::MAX, 0, types, true));
        }

        // Search root loop
        let compiled = Self::get_compiled(token);
        if let Some(found) = compiled
            .fail_descrs
            .iter()
            .find(|d| Arc::as_ptr(d) as usize == ptr)
        {
            return found.clone();
        }

        // Search bridge fail_descrs in asmmemmgr_blocks
        let blocks = token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                if let Some(found) = bridge
                    .fail_descrs
                    .iter()
                    .find(|d| Arc::as_ptr(d) as usize == ptr)
                {
                    return found.clone();
                }
            }
        }

        panic!(
            "find_descr_by_ptr: jf_descr {:#x} not found in root loop \
             or any bridge — RPython equivalent (AbstractDescr.show) \
             never fails",
            ptr
        );
    }

    /// Find a descr by (trace_id, fail_index) across root loop + all
    /// bridges. Used by compile_bridge to locate the exact guard descr
    /// that failed — RPython passes the faildescr object directly.
    ///
    /// trace_id == 0 is normalized to the root trace id, matching
    /// cranelift (compiler.rs:10092) and the BridgeFailDescrProxy
    /// convention.
    ///
    /// Panics if not found — in RPython, the faildescr is the exact
    /// object, so there is no lookup-miss path. Use `try_find_descr` for
    /// query-style callers (e.g. `bridge_was_compiled` /
    /// `compiled_bridge_fail_descr_layouts`) that legitimately probe
    /// "is this guard already compiled?" and must treat the miss as
    /// `None` (matching cranelift's `?`-on-miss semantics in
    /// `compiler.rs:11723`).
    fn find_descr(token: &JitCellToken, trace_id: u64, fail_index: u32) -> Arc<DynasmFailDescr> {
        Self::try_find_descr(token, trace_id, fail_index).unwrap_or_else(|| {
            panic!(
                "find_descr: (trace_id={}, fail_index={}) not found in \
                 root loop or any bridge — RPython uses exact faildescr \
                 object identity, so this lookup must succeed",
                trace_id, fail_index
            )
        })
    }

    fn try_find_descr(
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Arc<DynasmFailDescr>> {
        let compiled = Self::get_compiled(token);
        // Normalize trace_id: 0 → root trace id
        let trace_id = if trace_id == 0 {
            compiled.trace_id
        } else {
            trace_id
        };

        if let Some(found) = compiled
            .fail_descrs
            .iter()
            .find(|d| d.trace_id == trace_id && d.fail_index == fail_index)
        {
            return Some(found.clone());
        }
        let blocks = token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                if let Some(found) = bridge
                    .fail_descrs
                    .iter()
                    .find(|d| d.trace_id == trace_id && d.fail_index == fail_index)
                {
                    return Some(found.clone());
                }
            }
        }
        None
    }

    fn call_assembler_targets_snapshot() -> HashMap<u64, usize> {
        CALL_ASSEMBLER_TARGETS
            .lock()
            .expect("CALL_ASSEMBLER_TARGETS poisoned")
            .iter()
            .filter_map(|(&k, &addr)| if addr != 0 { Some((k, addr)) } else { None })
            .collect()
    }

    fn register_call_assembler_target(token_number: u64, code_addr: usize) {
        CALL_ASSEMBLER_TARGETS
            .lock()
            .expect("CALL_ASSEMBLER_TARGETS poisoned")
            .insert(token_number, code_addr);
    }

    fn redirect_call_assembler_target(old_number: u64, new_addr: usize) {
        CALL_ASSEMBLER_TARGETS
            .lock()
            .expect("CALL_ASSEMBLER_TARGETS poisoned")
            .insert(old_number, new_addr);
    }

    /// Static entry point for `lib.rs register_pending_call_assembler_target`.
    /// Inserts `code_addr = 0` (pending) so `call_assembler_targets_snapshot`
    /// filters it out; `compile_loop` overwrites with the real address.
    pub fn register_pending_call_assembler_target_static(token_number: u64) {
        CALL_ASSEMBLER_TARGETS
            .lock()
            .expect("CALL_ASSEMBLER_TARGETS poisoned")
            .entry(token_number)
            .or_insert(0);
    }

    /// `rpython/jit/backend/llsupport/llmodel.py:534-537`
    /// `get_baseofs_of_frame_field(self)` — offset from a `JITFRAME` base
    /// to the first frame-array item. Used by `_set_initial_bindings`
    /// (regalloc.py:865) and `update_frame_info` (model.py:316) for
    /// `jfi_frame_size` accounting (jitframe.py:19-22).
    fn get_baseofs_of_frame_field() -> i64 {
        crate::jitframe::FIRST_ITEM_OFFSET as i64
    }
}

impl Backend for DynasmBackend {
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;
        let header_pc = self.next_header_pc;

        // gc.py:109 rewrite_assembler parity: run GC rewriter before regalloc.
        let prepared_ops = self.prepare_ops_for_compile(inputargs, ops);
        let constants = std::mem::take(&mut self.constants);
        let typeid_table = self.collect_classptr_typeid_table(&prepared_ops, &constants);
        let mut asm = Asm::new(
            trace_id,
            header_pc,
            constants,
            self.vtable_offset,
            typeid_table,
        );
        asm.set_call_assembler_targets(Self::call_assembler_targets_snapshot());
        let compiled = asm.assemble_loop(inputargs, &prepared_ops)?;

        let code_addr = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let code_size = compiled.buffer.len();
        let frame_depth = compiled.frame_depth.load(Ordering::Acquire) as i64;
        Self::register_call_assembler_target(token.number, code_addr);

        // `rpython/jit/backend/x86/assembler.py:513-526` initializes the
        // per-loop `CompiledLoopToken` fields at assemble_loop entry:
        //   * frame_info is allocated and assigned (line 526-530)
        //   * looptoken.compiled_loop_token = clt (line 514)
        // pyre eagerly creates the CLT in `JitCellToken::new`, so the
        // equivalent here is populating its fields with the real values
        // computed during assembly.
        let baseofs = Self::get_baseofs_of_frame_field();
        if let Some(clt) = token.compiled_loop_token.as_ref() {
            // `x86/assembler.py:526-530` frame_info = malloc_aligned + set
            // jfi_frame_depth/jfi_frame_size. pyre's frame_info lives on
            // the CLT already; just populate via update_frame_depth.
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, frame_depth);
            // `llsupport/regalloc.py:861-871` `_set_initial_bindings` —
            // pyre lays inputargs at contiguous word slots in the frame
            // array, so loc.value - base_ofs = i * SIZEOFSIGNED for
            // inputarg i. The list length must match `inputargs.len()`
            // so `handle_call_assembler` (rewrite.py:673) can index it.
            let locs: Vec<i32> = (0..inputargs.len())
                .map(|i| (i as i32) * (crate::jitframe::SIZEOFSIGNED as i32))
                .collect();
            *clt._ll_initial_locs.lock() = locs;
        }
        // `x86/assembler.py:599` `looptoken._ll_function_addr =
        // rawstart + functionpos`. pyre stores the single entry point
        // so `_ll_function_addr` = compiled-code base.
        token._ll_function_addr = code_addr;
        token.compiled = Some(Box::new(compiled));

        Ok(AsmInfo {
            code_addr,
            code_size,
        })
    }

    fn set_constants(&mut self, constants: std::collections::HashMap<u32, i64>) {
        self.constants = constants;
    }

    fn set_constant_types(
        &mut self,
        constant_types: std::collections::HashMap<u32, majit_ir::Type>,
    ) {
        self.constant_types = constant_types;
    }

    fn set_next_trace_id(&mut self, trace_id: u64) {
        self.next_trace_id = trace_id;
    }

    fn set_next_header_pc(&mut self, header_pc: u64) {
        self.next_header_pc = header_pc;
    }

    fn set_done_with_this_frame_descr_void(&mut self, descr: majit_ir::DescrRef) {
        crate::guard::set_done_with_this_frame_descr_void(descr);
    }
    fn set_done_with_this_frame_descr_int(&mut self, descr: majit_ir::DescrRef) {
        crate::guard::set_done_with_this_frame_descr_int(descr);
    }
    fn set_done_with_this_frame_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        crate::guard::set_done_with_this_frame_descr_ref(descr);
    }
    fn set_done_with_this_frame_descr_float(&mut self, descr: majit_ir::DescrRef) {
        crate::guard::set_done_with_this_frame_descr_float(descr);
    }
    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        crate::guard::set_exit_frame_with_exception_descr_ref(descr);
    }
    fn set_propagate_exception_descr(&mut self, descr: majit_ir::DescrRef) {
        crate::guard::set_propagate_exception_descr(descr);
    }

    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &JitCellToken,
        _previous_tokens: &[JitCellToken],
    ) -> Result<AsmInfo, BackendError> {
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;

        let prepared_ops = self.prepare_ops_for_compile(inputargs, ops);
        let constants = std::mem::take(&mut self.constants);
        let typeid_table = self.collect_classptr_typeid_table(&prepared_ops, &constants);
        let mut asm = Asm::new(trace_id, 0, constants, self.vtable_offset, typeid_table);
        asm.set_call_assembler_targets(Self::call_assembler_targets_snapshot());

        let _orig_compiled = Self::get_compiled(original_token);

        let guard_descr = Self::find_descr(
            original_token,
            fail_descr.trace_id(),
            fail_descr.fail_index(),
        );
        let arglocs = Asm::rebuild_faillocs_from_descr(&guard_descr, inputargs);
        let compiled = asm.assemble_bridge(fail_descr, inputargs, &prepared_ops, &arglocs)?;

        let bridge_addr = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let code_size = compiled.buffer.len();

        // assembler.py:987 patch_jump_for_descr — redirect guard to bridge.
        // Use the exact guard descr found above, not a fail_index search.
        let ajo = guard_descr.adr_jump_offset();
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm-bridge] patch: trace_id={} fail_index={} adr_jump_offset=0x{:x} bridge_addr=0x{:x}",
                guard_descr.trace_id, guard_descr.fail_index, ajo, bridge_addr
            );
        }
        if ajo != 0 {
            Asm::patch_jump_for_descr(&guard_descr, bridge_addr);
        } else if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[dynasm-bridge] WARNING: adr_jump_offset=0, bridge NOT patched!");
        }
        guard_descr.set_bridge_addr(bridge_addr);

        // llmodel.py:252 asmmemmgr_blocks parity: store the entire
        // bridge CompiledCode on the owning loop token. This keeps
        // both the ExecutableBuffer (mapped code) AND the fail_descrs
        // (Arc<DynasmFailDescr>) alive. Recovery stubs embed raw
        // pointers to these Arcs — dropping them would create
        // dangling pointers when a bridge-internal guard fires.
        // RPython's asmmemmgr ties code blocks and their resume
        // descriptors to the same compiled_loop_token lifetime.
        original_token.asmmemmgr_blocks().push(Box::new(compiled));

        Ok(AsmInfo {
            code_addr: bridge_addr,
            code_size,
        })
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        // assembler.py:1080 `_call_header_with_stack_check` emits the
        // inline probe at the top of every compiled loop, matching
        // cranelift's `jit_prologue_stack_check_shim` call in
        // `compiler.rs:5868`. The prior runner-level `jit_prologue_stack_check`
        // call only guarded top-level entry and missed compiled-to-
        // compiled CALL_ASSEMBLER recursion.
        let compiled = Self::get_compiled(token);
        let entry = codebuf::buffer_ptr(&compiled.buffer);

        let num_slots = args
            .len()
            .max(compiled.fail_descrs.len() * 4)
            .max(compiled.frame_depth.load(Ordering::Acquire))
            .max(64);
        let jf_ptr = unsafe { libc::calloc(1, JitFrame::alloc_size(num_slots)) as *mut JitFrame };
        assert!(!jf_ptr.is_null(), "execute_token: calloc failed");
        unsafe { JitFrame::init(jf_ptr, std::ptr::null(), num_slots) };
        // Register this libc-allocated jitframe with the GC so its
        // interior Ref slots (pinned by gcmap bits) remain visible to
        // the collector during CallMallocNursery slow-path collections.
        majit_gc::shadow_stack::register_libc_jitframe(jf_ptr as usize);

        for (i, arg) in args.iter().enumerate() {
            let raw = match arg {
                Value::Int(v) => *v,
                Value::Ref(r) => r.0 as i64,
                Value::Float(f) => f.to_bits() as i64,
                Value::Void => 0,
            };
            unsafe { crate::llmodel::set_int_value(jf_ptr, Self::input_slot(i), raw as isize) };
        }

        if std::env::var_os("MAJIT_LOG").is_some() {
            for (i, arg) in args.iter().enumerate() {
                let raw = unsafe {
                    crate::llmodel::get_int_value_direct(jf_ptr, Self::input_slot(i)) as i64
                };
                eprintln!("[dynasm]   arg[{}] = {:#018x} ({:?})", i, raw as u64, arg);
            }
            eprintln!(
                "[dynasm] execute_token: entry={:?} jf_ptr={:?} num_args={} num_slots={} code_len={}",
                entry,
                jf_ptr,
                args.len(),
                num_slots,
                compiled.buffer.len()
            );
        }

        if std::env::var_os("MAJIT_DUMP").is_some() {
            let code = unsafe { std::slice::from_raw_parts(entry, compiled.buffer.len()) };
            eprintln!("[dynasm] CODE DUMP ({} bytes at {:?}):", code.len(), entry);
            for (i, chunk) in code.chunks(4).enumerate() {
                let word = u32::from_le_bytes([
                    chunk.get(0).copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                eprint!("{:08x} ", word);
                if (i + 1) % 8 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
        }

        // Debug: verify bridge patches are visible
        if std::env::var_os("MAJIT_LOG").is_some() {
            for descr in &compiled.fail_descrs {
                if descr.bridge_addr() != 0 && descr.adr_jump_offset() == 0 {
                    eprintln!(
                        "[dynasm] bridge-patched guard fi={} bridge_addr={:#x} ajo=0 (patched)",
                        descr.fail_index,
                        descr.bridge_addr()
                    );
                }
            }
        }

        // llmodel.py:323: ll_frame = func(ll_frame). The compiled
        // prologue (gen_shadowstack_header) / epilogue
        // (gen_footer_shadowstack) push/pop the jf_ptr onto the shadow
        // stack inline, matching aarch64/assembler.py:1422/1438 — no
        // manual push_jf/pop_jf_to around the call.
        let func: unsafe extern "C" fn(*mut JitFrame) -> *mut JitFrame =
            unsafe { std::mem::transmute(entry) };
        let result_jf = unsafe { func(jf_ptr) };

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] execute_token returned: result_jf={:?} (expected={:?}) same={}",
                result_jf,
                jf_ptr,
                result_jf == jf_ptr
            );
        }

        // llmodel.py:412-420 get_latest_descr: read jf_descr from frame.
        let jf_descr_raw = unsafe { crate::llmodel::get_latest_descr(result_jf) as i64 };
        let descr = Self::find_descr_by_ptr(token, jf_descr_raw as usize);

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] descr: fi={} finish={} types={} locs={:?}",
                descr.fail_index,
                descr.is_finish,
                descr.fail_arg_types.len(),
                &descr.fail_arg_locs
            );
        }

        // RPython parity: remap jitframe values using fail_arg_locs.
        let n_locs = descr.fail_arg_locs.len();
        let mut raw_values: Vec<i64> = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            if i < n_locs {
                match descr.fail_arg_locs[i] {
                    Some(slot) => {
                        let val =
                            unsafe { crate::llmodel::get_int_value_direct(result_jf, slot) as i64 };
                        if std::env::var_os("MAJIT_LOG").is_some() && i < 10 {
                            eprintln!(
                                "[dynasm] fail_arg[{}]: slot={} val={:#018x}",
                                i, slot, val as u64
                            );
                        }
                        raw_values.push(val);
                    }
                    None => raw_values.push(0),
                }
            } else {
                raw_values
                    .push(unsafe { crate::llmodel::get_int_value_direct(result_jf, i) as i64 });
            }
        }

        majit_gc::shadow_stack::unregister_libc_jitframe(jf_ptr as usize);
        unsafe { libc::free(jf_ptr as *mut std::ffi::c_void) };

        DeadFrame {
            data: Box::new(FrameData::new(raw_values, descr)),
        }
    }

    /// Override execute_token_ints_raw to return the FULL jitframe
    /// content (all slots), matching Cranelift's behavior.
    /// RPython: the deadframe IS the jitframe — all slots are accessible.
    fn execute_token_ints_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> majit_backend::RawExecResult {
        // Same rationale as `execute_token`: the inline probe emitted by
        // `_call_header` (x86/aarch64 assembler.rs) is now the sole
        // stack-overflow detection site, so no runner-level probe is
        // needed here.
        let compiled = Self::get_compiled(token);
        let entry = codebuf::buffer_ptr(&compiled.buffer);

        let num_slots = args
            .len()
            .max(compiled.fail_descrs.len() * 4)
            .max(compiled.frame_depth.load(Ordering::Acquire))
            .max(64);
        let jf_ptr = unsafe { libc::calloc(1, JitFrame::alloc_size(num_slots)) as *mut JitFrame };
        assert!(!jf_ptr.is_null(), "execute_token_ints_raw: calloc failed");
        unsafe { JitFrame::init(jf_ptr, std::ptr::null(), num_slots) };

        for (i, &val) in args.iter().enumerate() {
            unsafe { crate::llmodel::set_int_value(jf_ptr, Self::input_slot(i), val as isize) };
        }

        let func: unsafe extern "C" fn(*mut JitFrame) -> *mut JitFrame =
            unsafe { std::mem::transmute(entry) };
        let result_jf = unsafe { func(jf_ptr) };

        let jf_descr_raw = unsafe { crate::llmodel::get_latest_descr(result_jf) as i64 };
        let descr = Self::find_descr_by_ptr(token, jf_descr_raw as usize);

        let num_fail_args = descr.fail_arg_types.len();
        let mut outputs: Vec<i64> = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            outputs.push(unsafe { crate::llmodel::get_int_value_direct(result_jf, i) as i64 });
        }
        let mut typed_outputs = Vec::with_capacity(num_fail_args);
        for i in 0..num_fail_args {
            let raw = match descr.fail_arg_locs.get(i) {
                Some(Some(slot)) => outputs.get(*slot).copied().unwrap_or(0),
                Some(None) => 0,
                None => outputs.get(i).copied().unwrap_or(0),
            };
            typed_outputs.push(match descr.fail_arg_types[i] {
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                _ => Value::Int(raw),
            });
        }
        let exit_layout = Some(descr.layout());

        majit_gc::shadow_stack::unregister_libc_jitframe(jf_ptr as usize);
        unsafe { libc::free(jf_ptr as *mut std::ffi::c_void) };

        majit_backend::RawExecResult {
            outputs,
            typed_outputs,
            exit_layout,
            force_token_slots: Vec::new(),
            savedata: None,
            exception_value: GcRef::NULL,
            fail_index: descr.fail_index,
            trace_id: descr.trace_id,
            is_finish: descr.is_finish,
            status: descr.get_status(),
            descr_addr: Arc::as_ptr(&descr) as usize,
        }
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        let data = frame.data.downcast_ref::<FrameData>().unwrap();
        &*data.fail_descr
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        frame
            .data
            .downcast_ref::<FrameData>()
            .unwrap()
            .get_int(index)
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        frame
            .data
            .downcast_ref::<FrameData>()
            .unwrap()
            .get_float(index)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        frame
            .data
            .downcast_ref::<FrameData>()
            .unwrap()
            .get_ref(index)
    }

    fn invalidate_loop(&self, token: &JitCellToken) {
        token.invalidated.store(true, Ordering::Release);
    }

    // assembler.py:1138 redirect_call_assembler
    fn redirect_call_assembler(
        &self,
        old: &JitCellToken,
        new: &JitCellToken,
    ) -> Result<(), BackendError> {
        let old_compiled = Self::get_compiled(old);
        let new_compiled = Self::get_compiled(new);
        // x86/assembler.py:1146-1151 update_frame_info parity: propagate
        // new loop's frame depth onto the old token and every token in
        // its existing redirect chain, using the `baseofs` obtained from
        // `cpu.get_baseofs_of_frame_field()` so `jfi_frame_size` follows
        // jitframe.py:19-22 `base_ofs + new_depth * SIZEOFSIGNED`.
        let baseofs = Self::get_baseofs_of_frame_field();
        if let (Some(new_clt), Some(old_clt)) = (
            new.compiled_loop_token.as_ref(),
            old.compiled_loop_token.as_ref(),
        ) {
            // Seed new's CompiledLoopToken.frame_info.jfi_frame_depth
            // from the backend-specific compiled code depth so
            // update_frame_info has a non-zero value to propagate.
            let new_depth = new_compiled.frame_depth.load(Ordering::Acquire);
            new_clt
                .frame_info
                .lock()
                .update_frame_depth(baseofs, new_depth as i64);
            // model.py:316-329 update_frame_info — pass old CLT with a
            // weak ref for the "append self to chain" step (line 328
            // `new_loop_tokens.append(weakref.ref(oldlooptoken))`).
            let old_weak = Arc::downgrade(old_clt);
            new_clt.update_frame_info(old_clt, old_weak, baseofs);
            // Keep the backend-specific frame_depth in lockstep so bridge
            // codegen's existing readers (CompiledCode.frame_depth) also
            // see the propagated value. PRE-EXISTING-ADAPTATION: RPython
            // reads the depth back from `compiled_loop_token.frame_info`;
            // dynasm's codegen reads `CompiledCode.frame_depth`. Writing
            // both keeps the orthodox field authoritative while the
            // reader migration lands.
            old_compiled
                .frame_depth
                .fetch_max(new_depth, Ordering::Release);
        }
        let old_addr = codebuf::buffer_ptr(&old_compiled.buffer);
        let new_addr = codebuf::buffer_ptr(&new_compiled.buffer);
        Asm::redirect_call_assembler(old_addr, new_addr);
        Self::redirect_call_assembler_target(old.number, new_addr as usize);
        Ok(())
    }

    // No migrate_bridges — we patch in place.

    fn store_guard_hashes(&self, token: &JitCellToken, hashes: &[u64]) {
        let compiled = Self::get_compiled(token);
        for (i, &hash) in hashes.iter().enumerate() {
            if let Some(descr) = compiled.fail_descrs.get(i) {
                if !descr.is_finish && descr.get_status() == 0 {
                    descr.store_hash(hash);
                }
            }
        }
    }

    fn get_guard_status(
        &self,
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
    ) -> (u64, usize) {
        let descr = Self::find_descr(token, trace_id, fail_index);
        (descr.get_status(), Arc::as_ptr(&descr) as usize)
    }

    fn store_bridge_guard_hashes(
        &self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
        hashes: &[u64],
    ) {
        let source_descr = Self::find_descr(token, source_trace_id, source_fail_index);
        let bridge_addr = source_descr.bridge_addr();
        if bridge_addr == 0 {
            return;
        }
        let blocks = token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                let addr = codebuf::buffer_ptr(&bridge.buffer) as usize;
                if addr == bridge_addr {
                    for (i, &hash) in hashes.iter().enumerate() {
                        if let Some(descr) = bridge.fail_descrs.get(i) {
                            if !descr.is_finish && descr.get_status() == 0 {
                                descr.store_hash(hash);
                            }
                        }
                    }
                    return;
                }
            }
        }
    }

    fn read_descr_status(&self, descr_addr: usize) -> u64 {
        let descr = unsafe { &*(descr_addr as *const DynasmFailDescr) };
        descr.get_status()
    }

    fn start_compiling_descr(&self, descr_addr: usize) {
        let descr = unsafe { &*(descr_addr as *const DynasmFailDescr) };
        descr.start_compiling();
    }

    fn done_compiling_descr(&self, descr_addr: usize) {
        let descr = unsafe { &*(descr_addr as *const DynasmFailDescr) };
        descr.done_compiling();
    }

    fn bh_new(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        let ptr = unsafe { libc::malloc(size) };
        if !ptr.is_null() {
            unsafe { libc::memset(ptr, 0, size) };
        }
        ptr as i64
    }

    fn bh_new_with_vtable(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        let vtable = sizedescr.get_vtable();
        let ptr = unsafe { libc::malloc(size) };
        if !ptr.is_null() {
            unsafe {
                libc::memset(ptr, 0, size);
                // llmodel.py:780-782: if self.vtable_offset is not None:
                //   self.write_int_at_mem(res, self.vtable_offset, WORD, sizedescr.get_vtable())
                if let Some(vt_off) = self.vtable_offset {
                    if vtable != 0 {
                        *((ptr as *mut u8).add(vt_off) as *mut usize) = vtable;
                    }
                }
            }
        }
        ptr as i64
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer through the installed gc_ll_descr.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        self.lookup_typeid_from_classptr(classptr)
    }

    /// llmodel.py:747-750 bh_raw_load_i(addr, offset, descr).
    fn bh_raw_load_i(
        &self,
        addr: i64,
        offset: i64,
        descr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        // llmodel.py:748-749: ofs, size, sign = self.unpack_arraydescr_size(descr)
        // ofs == 0 always for raw lengthless arrays (llmodel.py:749 assert)
        let size = descr.as_itemsize();
        let sign = descr.is_item_signed();
        // llmodel.py:750: return self.read_int_at_mem(addr, offset, size, sign)
        self.read_int_at_mem(addr, offset, size, sign)
    }

    /// llmodel.py:739-742 bh_raw_store_i(addr, offset, newvalue, descr).
    fn bh_raw_store_i(
        &self,
        addr: i64,
        offset: i64,
        newvalue: i64,
        descr: &majit_translate::jitcode::BhDescr,
    ) {
        // llmodel.py:740-741: ofs, size, _ = self.unpack_arraydescr_size(descr)
        // ofs == 0 always for raw lengthless arrays (llmodel.py:741 assert)
        let size = descr.as_itemsize();
        // llmodel.py:742: self.write_int_at_mem(addr, offset, size, newvalue)
        self.write_int_at_mem(addr, offset, size, newvalue);
    }

    /// llmodel.py:752-753 bh_raw_load_f(addr, offset, descr).
    fn bh_raw_load_f(
        &self,
        addr: i64,
        offset: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        // llmodel.py:753: return self.read_float_at_mem(addr, offset)
        self.read_float_at_mem(addr, offset)
    }

    /// llmodel.py:744-745 bh_raw_store_f(addr, offset, newvalue, descr).
    fn bh_raw_store_f(
        &self,
        addr: i64,
        offset: i64,
        newvalue: f64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
        // llmodel.py:745: self.write_float_at_mem(addr, offset, newvalue)
        self.write_float_at_mem(addr, offset, newvalue);
    }

    fn bh_getfield_gc_i(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        let offset = fielddescr.as_offset();
        unsafe { *((struct_ptr as *const u8).add(offset) as *const i64) }
    }

    fn bh_getfield_gc_r(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        let offset = fielddescr.as_offset();
        GcRef(unsafe { *((struct_ptr as *const u8).add(offset) as *const usize) })
    }

    fn bh_setfield_gc_i(
        &self,
        struct_ptr: i64,
        value: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
        let offset = fielddescr.as_offset();
        unsafe { *((struct_ptr as *mut u8).add(offset) as *mut i64) = value };
    }

    fn bh_setfield_gc_r(
        &self,
        struct_ptr: i64,
        value: GcRef,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
        let offset = fielddescr.as_offset();
        unsafe { *((struct_ptr as *mut u8).add(offset) as *mut usize) = value.0 };
    }

    /// compile_tmp_callback parity: register a placeholder for a pending
    /// CALL_ASSEMBLER target. The real code_addr is set by compile_loop.
    /// Until then, CALL_ASSEMBLER's generated code falls through to the
    /// helper trampoline which calls force_fn (interpreter re-execution).
    fn register_pending_target(
        &mut self,
        token_number: u64,
        _input_types: Vec<Type>,
        _num_inputs: usize,
        _num_scalar_inputargs: usize,
        _index_of_virtualizable: i32,
    ) {
        // Insert code_addr = 0 (pending). call_assembler_targets_snapshot
        // excludes pending entries, so the generated CALL_ASSEMBLER code
        // takes the "unresolved target" path → helper trampoline →
        // force_fn. `compile_loop` overwrites with the real code_addr.
        //
        // The typed-metadata args (`num_inputs`, `num_scalar_inputargs`,
        // `index_of_virtualizable`) are the cranelift backend's API —
        // dynasm does not consume them because the address registry is
        // purely a `token_number -> _ll_function_addr` index. RPython
        // parity: `x86/assembler.py:599` stores `_ll_function_addr` on
        // the looptoken; pyre serializes the token by number, so this
        // HashMap is the minimum adaptation needed. Orthodox metadata
        // (`_ll_initial_locs`, `frame_info`) lives on
        // `token.compiled_loop_token` (`model.py:293-294`).
        Self::register_pending_call_assembler_target_static(token_number);
    }

    fn compiled_fail_descr_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = Self::get_compiled(token);
        Some(compiled.fail_descrs.iter().map(|d| d.layout()).collect())
    }

    fn compiled_trace_fail_descr_layouts(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = Self::get_compiled(token);
        if compiled.trace_id == trace_id {
            return Some(compiled.fail_descrs.iter().map(|d| d.layout()).collect());
        }
        // Search bridge fail_descrs in asmmemmgr_blocks.
        let blocks = token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                if bridge.trace_id == trace_id {
                    return Some(bridge.fail_descrs.iter().map(|d| d.layout()).collect());
                }
            }
        }
        None
    }

    fn compiled_bridge_fail_descr_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        // RPython faildescr lookup is by object identity, never misses.
        // majit query-style callers (`bridge_was_compiled` etc.) probe
        // by (trace_id, fail_index) and must treat the miss as `None`
        // — match cranelift's `?` semantics in
        // `compiler.rs:11723 compiled_bridge_fail_descr_layouts`.
        let source_descr =
            Self::try_find_descr(original_token, source_trace_id, source_fail_index)?;
        let bridge_addr = source_descr.bridge_addr();
        if bridge_addr == 0 {
            return None;
        }
        let blocks = original_token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                let addr = codebuf::buffer_ptr(&bridge.buffer) as usize;
                if addr == bridge_addr {
                    return Some(bridge.fail_descrs.iter().map(|d| d.layout()).collect());
                }
            }
        }
        None
    }

    fn update_fail_descr_recovery_layout(
        &mut self,
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
        recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        let descr = Self::find_descr(token, trace_id, fail_index);
        descr.set_recovery_layout(recovery_layout);
        true
    }

    fn setup_once(&mut self) {}
    fn finish_once(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_backend::Backend;
    use majit_gc::collector::MiniMarkGC;
    use majit_gc::trace::TypeInfo;

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// Verify the dynasm backend's gc_ll_descr round-trips a registered
    /// vtable→type_id mapping (the same contract Cranelift uses).
    #[test]
    fn test_backend_typeid_from_classptr_via_gc_ll_descr() {
        let mut gc = MiniMarkGC::new();
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let int_vtable: usize = 0x2222_3300;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let resolved = backend.get_typeid_from_classptr_if_gcremovetypeptr(int_vtable);
        assert_eq!(resolved, Some(int_tid));
        let unknown = backend.get_typeid_from_classptr_if_gcremovetypeptr(0xCAFE_F00D);
        assert_eq!(unknown, None);
    }

    #[test]
    fn test_backend_installs_active_gc_guard_hooks() {
        let mut gc = MiniMarkGC::new();
        let obj_tid = gc.register_type(TypeInfo::object(16));
        let obj = gc.alloc_with_type(obj_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        assert!(majit_gc::supports_guard_gc_type());
        assert!(majit_gc::check_is_object(obj));
        assert_eq!(majit_gc::get_actual_typeid(obj), Some(obj_tid));
        assert_eq!(majit_gc::typeid_is_object(obj_tid), Some(true));
    }
}

// ── rewrite.py:489 parity: inject str_descr/unicode_descr ──

const BUILTIN_STRING_LEN_OFFSET: usize = std::mem::size_of::<usize>();
const BUILTIN_STRING_BASE_SIZE: usize = BUILTIN_STRING_LEN_OFFSET + std::mem::size_of::<usize>();

#[derive(Debug)]
struct BuiltinFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
}

impl majit_ir::Descr for BuiltinFieldDescr {
    fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self)
    }
}

impl majit_ir::FieldDescr for BuiltinFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }
    fn field_size(&self) -> usize {
        self.field_size
    }
    fn field_type(&self) -> Type {
        self.field_type
    }
    fn is_field_signed(&self) -> bool {
        self.signed
    }
}

#[derive(Debug)]
struct BuiltinArrayDescr {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_descr: Arc<BuiltinFieldDescr>,
}

impl majit_ir::Descr for BuiltinArrayDescr {
    fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
        Some(self)
    }
}

impl majit_ir::ArrayDescr for BuiltinArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }
    fn item_size(&self) -> usize {
        self.item_size
    }
    fn type_id(&self) -> u32 {
        self.type_id
    }
    fn item_type(&self) -> Type {
        self.item_type
    }
    fn is_item_signed(&self) -> bool {
        self.signed
    }
    fn len_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self.len_descr.as_ref())
    }
}

fn builtin_string_array_descr(opcode: majit_ir::OpCode) -> Option<majit_ir::DescrRef> {
    use majit_ir::OpCode;
    let item_size = match opcode {
        OpCode::Newstr
        | OpCode::Strlen
        | OpCode::Strgetitem
        | OpCode::Strsetitem
        | OpCode::Copystrcontent
        | OpCode::Strhash => 1,
        OpCode::Newunicode
        | OpCode::Unicodelen
        | OpCode::Unicodegetitem
        | OpCode::Unicodesetitem
        | OpCode::Copyunicodecontent
        | OpCode::Unicodehash => 4,
        _ => return None,
    };
    let len_descr = Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_LEN_OFFSET,
        field_size: 8,
        field_type: Type::Int,
        signed: false,
    });
    Some(Arc::new(BuiltinArrayDescr {
        base_size: BUILTIN_STRING_BASE_SIZE,
        item_size,
        type_id: 0,
        item_type: Type::Int,
        signed: false,
        len_descr,
    }))
}

fn inject_builtin_string_descrs(ops: &mut [Op]) {
    for op in ops {
        if op.descr.is_none() {
            if let Some(descr) = builtin_string_array_descr(op.opcode) {
                op.descr = Some(descr);
            }
        }
    }
}
