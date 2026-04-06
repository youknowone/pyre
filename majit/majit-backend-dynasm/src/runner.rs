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
}

impl DynasmBackend {
    pub fn new() -> Self {
        DynasmBackend {
            next_trace_id: 1,
            next_header_pc: 0,
            constants: std::collections::HashMap::new(),
        }
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

    /// Stub — GC allocator registration.
    pub fn set_gc_allocator(&mut self, _gc: Box<dyn majit_gc::GcAllocator>) {}

    /// llmodel.py:64-69 self.vtable_offset — stub for the dynasm backend.
    pub fn set_vtable_offset(&mut self, _offset: Option<usize>) {}

    fn get_compiled(token: &JitCellToken) -> &CompiledCode {
        token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledCode>()
            .expect("compiled data is not CompiledCode")
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
        let asm = Assembler386::new(trace_id, header_pc, constants);
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
        let asm = Assembler386::new(trace_id, 0, constants);
        let compiled = asm.assemble_bridge(fail_descr, inputargs, ops)?;

        let bridge_addr = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let code_size = compiled.buffer.len();

        // assembler.py:987 patch_jump_for_descr — redirect guard to bridge.
        // Find the DynasmFailDescr in the original token and patch it.
        let orig_compiled = Self::get_compiled(original_token);
        if let Some(descr) = orig_compiled
            .fail_descrs
            .get(fail_descr.fail_index() as usize)
        {
            if descr.adr_jump_offset() != 0 {
                Assembler386::patch_jump_for_descr(descr, bridge_addr);
            }
            descr.set_bridge_addr(bridge_addr);
        }

        // Note: the bridge's CompiledCode must stay alive (ExecutableBuffer
        // keeps the memory mapped). We store it on the descr or a side table.
        // TODO: proper bridge lifetime management

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
        let num_slots = args.len().max(compiled.fail_descrs.len() * 4).max(64);
        let jf_total = 1 + num_slots; // 1 for jf_descr
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

        // llmodel.py:412-420 get_latest_descr: read jf_descr from frame
        let jf_descr_raw = unsafe { *result_jf.add(0) };

        // Find the matching fail descriptor by pointer
        let descr = compiled
            .fail_descrs
            .iter()
            .find(|d| Arc::as_ptr(d) as usize == jf_descr_raw as usize)
            .cloned()
            .unwrap_or_else(|| {
                compiled
                    .fail_descrs
                    .last()
                    .cloned()
                    .unwrap_or_else(|| Arc::new(DynasmFailDescr::new(0, 0, Vec::new(), true)))
            });

        // RPython: the deadframe IS the jitframe — ALL slots accessible.
        // Extract full jitframe content so rd_numb can reference any slot.
        let mut raw_values = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            raw_values.push(unsafe { *result_jf.add(1 + i) });
        }

        // Create a descriptor with fail_arg_types covering ALL slots.
        // RPython's get_value_direct accesses any jf_frame position.
        let full_types: Vec<Type> = (0..num_slots)
            .map(|i| descr.fail_arg_types.get(i).copied().unwrap_or(Type::Int))
            .collect();
        let full_descr = Arc::new(DynasmFailDescr::new(
            descr.fail_index,
            descr.trace_id,
            full_types,
            descr.is_finish,
        ));

        std::mem::forget(jf);

        DeadFrame {
            data: Box::new(FrameData::new(raw_values, full_descr)),
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

        let num_slots = args.len().max(compiled.fail_descrs.len() * 4).max(64);
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
        let descr = compiled
            .fail_descrs
            .iter()
            .find(|d| Arc::as_ptr(d) as usize == jf_descr_raw as usize)
            .cloned()
            .unwrap_or_else(|| {
                compiled
                    .fail_descrs
                    .last()
                    .cloned()
                    .unwrap_or_else(|| Arc::new(DynasmFailDescr::new(0, 0, Vec::new(), true)))
            });

        // Extract ALL values — RPython returns the entire jitframe.
        let mut outputs = Vec::with_capacity(num_slots);
        let mut typed_outputs = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            let raw = unsafe { *result_jf.add(1 + i) };
            outputs.push(raw);
            // Infer type from fail_arg_types if within range, else Int.
            let tp = descr.fail_arg_types.get(i).copied().unwrap_or(Type::Int);
            typed_outputs.push(match tp {
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                _ => Value::Int(raw),
            });
        }

        // Build exit_layout with ALL slots as type info.
        // RPython: deadframe contains entire jitframe, not just fail_args.
        let exit_types: Vec<Type> = (0..num_slots)
            .map(|i| descr.fail_arg_types.get(i).copied().unwrap_or(Type::Int))
            .collect();
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

        std::mem::forget(jf);

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
                *(ptr as *mut usize) = vtable;
            }
        }
        ptr as i64
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
