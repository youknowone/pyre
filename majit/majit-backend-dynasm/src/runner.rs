use std::sync::Arc;
/// runner.py: AbstractX86CPU — the Backend trait implementation.
///
/// This is the entry point for the dynasm backend, corresponding to
/// rpython/jit/backend/x86/runner.py AbstractX86CPU.
use std::sync::atomic::Ordering;

use majit_backend::{
    AsmInfo, Backend, BackendError, DeadFrame, ExitRecoveryLayout, FailDescrLayout, JitCellToken,
    RawExecResult, TerminalExitLayout,
};
use majit_ir::{FailDescr, GcRef, InputArg, Op, Type, Value};

use crate::assembler::{Assembler386, CompiledCode};
use crate::codebuf;
use crate::frame::FrameData;
use crate::guard::DynasmFailDescr;

/// runner.py:23 AbstractX86CPU — concrete Backend implementation.
pub struct DynasmBackend {
    /// Next unique trace ID.
    next_trace_id: u64,
    /// Next header PC (green key).
    next_header_pc: u64,
    /// Constants for the next compilation.
    constants: std::collections::HashMap<u32, i64>,
    /// llmodel.py:64-69 self.vtable_offset — byte offset of the typeptr
    /// field inside instance objects. None when gcremovetypeptr is enabled.
    vtable_offset: Option<usize>,
    /// llmodel.py self.gc_ll_descr — GC layer used by bh_new and the
    /// gcremovetypeptr branch of `_cmp_guard_class` to look up typeids.
    gc_ll_descr: Option<Box<dyn majit_gc::GcAllocator>>,
}

impl DynasmBackend {
    pub fn new() -> Self {
        DynasmBackend {
            next_trace_id: 1,
            next_header_pc: 0,
            constants: std::collections::HashMap::new(),
            vtable_offset: None,
            gc_ll_descr: None,
        }
    }

    /// Active vtable_offset for the assembler to consume during codegen.
    pub fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    /// Set constants for the next compile_loop/compile_bridge call.
    pub fn set_constants(&mut self, constants: std::collections::HashMap<u32, i64>) {
        self.constants = constants;
    }

    /// Force the next compile to use a specific trace id.
    pub fn set_next_trace_id(&mut self, trace_id: u64) {
        self.next_trace_id = trace_id;
    }

    /// Set the green_key (header PC) for the next compilation.
    pub fn set_next_header_pc(&mut self, header_pc: u64) {
        self.next_header_pc = header_pc;
    }

    /// Stub — dynasm doesn't need GC runtime ID.
    pub fn gc_runtime_id(&self) -> Option<u64> {
        None
    }
    pub fn set_gc_runtime_id(&mut self, _id: u64) {}

    /// llmodel.py:53-54: store gc_ll_descr on the cpu instance.
    /// dynasm has no allocator codegen yet, but the gc_ll_descr is still
    /// needed by the gcremovetypeptr branch of _cmp_guard_class.
    pub fn set_gc_allocator(&mut self, gc: Box<dyn majit_gc::GcAllocator>) {
        self.gc_ll_descr = Some(gc);
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
        self.gc_ll_descr
            .as_ref()
            .and_then(|gc: &Box<dyn majit_gc::GcAllocator>| {
                gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr)
            })
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
        if self.vtable_offset.is_some() || self.gc_ll_descr.is_none() {
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

    /// llmodel.py:412 get_latest_descr parity: resolve a raw jf_descr
    /// pointer to its Arc<DynasmFailDescr>. Searches root loop
    /// fail_descrs first, then all bridge fail_descrs stored in
    /// asmmemmgr_blocks. RPython does this via AbstractDescr.show()
    /// which works for any descr from any loop/bridge.
    ///
    /// Panics if not found — RPython uses object identity, so lookup
    /// failure is impossible in well-formed execution.
    fn find_descr_by_ptr(token: &JitCellToken, ptr: usize) -> Arc<DynasmFailDescr> {
        // compile.py:665-674 done_with_this_frame_descr singleton
        let global_done = crate::guard::done_with_this_frame_descr_ptr();
        if ptr == global_done {
            return Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![Type::Int], true));
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
        let blocks = token.asmmemmgr_blocks.lock();
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
    /// object, so there is no lookup-miss path.
    fn find_descr(token: &JitCellToken, trace_id: u64, fail_index: u32) -> Arc<DynasmFailDescr> {
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
            return found.clone();
        }
        let blocks = token.asmmemmgr_blocks.lock();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                if let Some(found) = bridge
                    .fail_descrs
                    .iter()
                    .find(|d| d.trace_id == trace_id && d.fail_index == fail_index)
                {
                    return found.clone();
                }
            }
        }

        panic!(
            "find_descr: (trace_id={}, fail_index={}) not found in \
             root loop or any bridge — RPython uses exact faildescr \
             object identity, so this lookup must succeed",
            trace_id, fail_index
        );
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

        let constants = std::mem::take(&mut self.constants);
        let typeid_table = self.collect_classptr_typeid_table(ops, &constants);
        let asm = Assembler386::new(
            trace_id,
            header_pc,
            constants,
            self.vtable_offset,
            typeid_table,
        );
        let compiled = asm.assemble_loop(inputargs, ops)?;

        let code_addr = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let code_size = compiled.buffer.len();
        token.compiled = Some(Box::new(compiled));

        Ok(AsmInfo {
            code_addr,
            code_size,
        })
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

        let constants = std::mem::take(&mut self.constants);
        let typeid_table = self.collect_classptr_typeid_table(ops, &constants);
        let mut asm = Assembler386::new(trace_id, 0, constants, self.vtable_offset, typeid_table);

        // assembler.py closing_jump parity: set the bridge's JUMP target
        let orig_compiled = Self::get_compiled(original_token);
        if orig_compiled.label_addr != 0 {
            asm.set_jump_target_addr(orig_compiled.label_addr);
        }

        // assembler.py:638 rebuild_faillocs_from_descr(faildescr, ...)
        // RPython uses the exact faildescr object. We find the matching
        // DynasmFailDescr by (trace_id, fail_index) across root loop +
        // all bridges. This handles bridge-on-bridge correctly.
        let guard_descr = Self::find_descr(
            original_token,
            fail_descr.trace_id(),
            fail_descr.fail_index(),
        );
        let source_slots = guard_descr.fail_args_slots.clone();
        let compiled = asm.assemble_bridge(fail_descr, inputargs, ops, &source_slots)?;

        let bridge_addr = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let code_size = compiled.buffer.len();

        // assembler.py:987 patch_jump_for_descr — redirect guard to bridge.
        // Use the exact guard descr found above, not a fail_index search.
        if guard_descr.adr_jump_offset() != 0 {
            Assembler386::patch_jump_for_descr(&guard_descr, bridge_addr);
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
        original_token
            .asmmemmgr_blocks
            .lock()
            .push(Box::new(compiled));

        Ok(AsmInfo {
            code_addr: bridge_addr,
            code_size,
        })
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let compiled = Self::get_compiled(token);
        let entry = codebuf::buffer_ptr(&compiled.buffer);

        // llmodel.py:298 malloc_jitframe → allocate jitframe
        // jf_ptr layout: [jf_descr, jf_frame[0], jf_frame[1], ...]
        let num_slots = args
            .len()
            .max(compiled.fail_descrs.len() * 4)
            .max(compiled.frame_depth.load(Ordering::Acquire))
            .max(64);
        let jf_total = 1 + num_slots;
        let mut jf: Vec<i64> = vec![0i64; jf_total];
        let jf_ptr = jf.as_mut_ptr();

        // Copy input values to jf_frame[0..n]
        for (i, arg) in args.iter().enumerate() {
            let raw = match arg {
                Value::Int(v) => *v,
                Value::Ref(r) => r.0 as i64,
                Value::Float(f) => f.to_bits() as i64,
                Value::Void => 0,
            };
            unsafe { *jf_ptr.add(1 + i) = raw };
        }

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] execute_token: entry={:?} jf_ptr={:?} num_args={} num_slots={} code_len={}",
                entry,
                jf_ptr,
                args.len(),
                num_slots,
                compiled.buffer.len()
            );
        }

        // llmodel.py:323: ll_frame = func(ll_frame)
        let func: unsafe extern "C" fn(*mut i64) -> *mut i64 =
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
        // RPython resolves jf_descr via AbstractDescr.show() which
        // works for any descr from any loop or bridge. We search root
        // loop fail_descrs + all bridge fail_descrs in asmmemmgr_blocks.
        let jf_descr_raw = unsafe { *result_jf.add(0) };
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
        // Position i in raw_values holds the value of fail_args[i],
        // read from its actual jitframe slot via fail_arg_locs[i].
        // This matches what RPython's _push_all_regs_to_frame +
        // _update_at_exit produce.
        let n_locs = descr.fail_arg_locs.len();
        let mut raw_values = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            if i < n_locs {
                match descr.fail_arg_locs[i] {
                    Some(slot) => raw_values.push(unsafe { *result_jf.add(1 + slot) }),
                    None => raw_values.push(0), // virtual: no physical value
                }
            } else {
                raw_values.push(unsafe { *result_jf.add(1 + i) });
            }
        }

        drop(jf);

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
        let compiled = Self::get_compiled(token);
        let entry = codebuf::buffer_ptr(&compiled.buffer);

        let num_slots = args
            .len()
            .max(compiled.fail_descrs.len() * 4)
            .max(compiled.frame_depth.load(Ordering::Acquire))
            .max(64);
        let jf_total = 1 + num_slots;
        let mut jf: Vec<i64> = vec![0i64; jf_total];
        let jf_ptr = jf.as_mut_ptr();

        for (i, &val) in args.iter().enumerate() {
            unsafe { *jf_ptr.add(1 + i) = val };
        }

        let func: unsafe extern "C" fn(*mut i64) -> *mut i64 =
            unsafe { std::mem::transmute(entry) };
        let result_jf = unsafe { func(jf_ptr) };

        let jf_descr_raw = unsafe { *result_jf.add(0) };
        let descr = Self::find_descr_by_ptr(token, jf_descr_raw as usize);

        // RPython parity: typed_outputs correspond to fail_arg_types.
        // raw outputs contain ALL jitframe slots; typed_outputs only
        // include the fail_arg entries (matching Cranelift behavior).
        let num_fail_args = descr.fail_arg_types.len();
        let mut outputs = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            outputs.push(unsafe { *result_jf.add(1 + i) });
        }
        let mut typed_outputs = Vec::with_capacity(num_fail_args);
        for i in 0..num_fail_args {
            let raw = match descr.fail_arg_locs.get(i) {
                Some(Some(slot)) => outputs.get(*slot).copied().unwrap_or(0),
                Some(None) => 0, // virtual
                None => outputs.get(i).copied().unwrap_or(0),
            };
            typed_outputs.push(match descr.fail_arg_types[i] {
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                _ => Value::Int(raw),
            });
        }
        let exit_types = descr.fail_arg_types.clone();
        let exit_layout = Some(majit_backend::FailDescrLayout {
            fail_index: descr.fail_index,
            fail_arg_types: exit_types,
            is_finish: descr.is_finish,
            trace_id: descr.trace_id,
            source_op_index: None,
            gc_ref_slots: Vec::new(),
            force_token_slots: Vec::new(),
            frame_stack: None,
            recovery_layout: None,
            trace_info: None,
        });

        drop(jf);

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
        // assembler.py:1148-1151 update_frame_info parity: propagate
        // new loop's frame_depth to old token so that callers entering
        // via the old token allocate a jitframe large enough for the
        // new (potentially deeper) loop body.
        let new_depth = new_compiled.frame_depth.load(Ordering::Acquire);
        old_compiled
            .frame_depth
            .fetch_max(new_depth, Ordering::Release);
        let old_addr = codebuf::buffer_ptr(&old_compiled.buffer);
        let new_addr = codebuf::buffer_ptr(&new_compiled.buffer);
        Assembler386::redirect_call_assembler(old_addr, new_addr);
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

    fn bh_new(&self, sizedescr: &dyn majit_ir::SizeDescr) -> i64 {
        let size = sizedescr.size();
        let ptr = unsafe { libc::malloc(size) };
        if !ptr.is_null() {
            unsafe { libc::memset(ptr, 0, size) };
        }
        ptr as i64
    }

    fn bh_new_with_vtable(&self, sizedescr: &dyn majit_ir::SizeDescr) -> i64 {
        let size = sizedescr.size();
        let vtable = sizedescr.vtable();
        let ptr = unsafe { libc::malloc(size) };
        if !ptr.is_null() {
            unsafe {
                libc::memset(ptr, 0, size);
                // llmodel.py:778-782 bh_new_with_vtable: write_int_at_mem(
                //     res, self.vtable_offset, WORD, sizedescr.get_vtable())
                if let Some(off) = self.vtable_offset {
                    *((ptr as *mut u8).add(off) as *mut usize) = vtable;
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

    fn bh_setfield_gc_i(&self, struct_ptr: i64, offset: usize, value: i64) {
        unsafe { *((struct_ptr as *mut u8).add(offset) as *mut i64) = value };
    }

    fn bh_setfield_gc_r(&self, struct_ptr: i64, offset: usize, value: GcRef) {
        unsafe { *((struct_ptr as *mut u8).add(offset) as *mut usize) = value.0 };
    }

    fn bh_getfield_gc_i(&self, struct_ptr: i64, offset: usize) -> i64 {
        unsafe { *((struct_ptr as *const u8).add(offset) as *const i64) }
    }

    fn bh_getfield_gc_r(&self, struct_ptr: i64, offset: usize) -> GcRef {
        GcRef(unsafe { *((struct_ptr as *const u8).add(offset) as *const usize) })
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
}
