/// WebAssembly backend for majit.
///
/// Generates wasm bytecodes via wasm-encoder. On wasm32 targets,
/// instantiates modules via JS WebAssembly.instantiate() and executes
/// via JS interop. On native targets, compile_loop succeeds but
/// execute_token requires the JS runtime (unreachable natively).
pub mod codegen;
pub mod failguard;

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
mod js_glue;

use std::collections::HashMap;
use std::sync::Arc;

use failguard::{CompiledWasmLoop, WasmFailDescr, WasmFrameData};
use majit_backend::{AsmInfo, BackendError, DeadFrame, JitCellToken};
use majit_ir::{FailDescr, GcRef, InputArg, Op, OpRef, Type, Value};

pub struct WasmBackend {
    trace_counter: u64,
    /// Optimizer constant pool (OpRef >= CONST_BASE → i64 value).
    constants: HashMap<u32, i64>,
}

impl WasmBackend {
    pub fn new() -> Self {
        WasmBackend {
            trace_counter: 0,
            constants: HashMap::new(),
        }
    }

    /// Set the constant pool (CraneliftBackend parity).
    pub fn set_constants(&mut self, constants: HashMap<u32, i64>) {
        self.constants = constants;
    }

    /// Set the next trace ID (CraneliftBackend parity).
    pub fn set_next_trace_id(&mut self, trace_id: u64) {
        self.trace_counter = trace_id;
    }

    /// Set the header PC (CraneliftBackend parity — no-op for wasm).
    pub fn set_next_header_pc(&mut self, _header_pc: u64) {}

    /// Set GC allocator (CraneliftBackend parity — no-op for wasm).
    pub fn set_gc_allocator(&mut self, _gc: Box<dyn majit_gc::GcAllocator>) {}

    /// Collect constants from ops (constant OpRefs that appear as args).
    fn collect_constants_from_ops(&mut self, ops: &[Op]) {
        for op in ops {
            for &arg in &op.args {
                if arg.is_constant() && !self.constants.contains_key(&arg.0) {
                    // Default to 0 if not already registered
                    self.constants.insert(arg.0, 0);
                }
            }
            if let Some(ref fail_args) = op.fail_args {
                for &arg in fail_args.iter() {
                    if arg.is_constant() && !self.constants.contains_key(&arg.0) {
                        self.constants.insert(arg.0, 0);
                    }
                }
            }
        }
    }
}

unsafe impl Send for WasmBackend {}

impl majit_backend::Backend for WasmBackend {
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;

        let (wasm_bytes, guard_exits) = codegen::build_wasm_module(inputargs, ops, &self.constants);

        // Build fail descriptors
        let fail_descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .map(|g| {
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    is_finish: g.is_finish,
                })
            })
            .collect();

        let max_output_slots = guard_exits
            .iter()
            .map(|g| g.fail_arg_refs.len())
            .max()
            .unwrap_or(0)
            .max(inputargs.len());

        // Compile via JS on wasm32, or store bytes for testing on native
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        let func_handle = js_glue::compile_module(&wasm_bytes);
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        let func_handle = 0u32; // Placeholder — no JS runtime available

        let compiled = CompiledWasmLoop {
            trace_id,
            input_types: inputargs.iter().map(|ia| ia.tp).collect(),
            func_handle,
            fail_descrs,
            num_inputs: inputargs.len(),
            max_output_slots,
        };

        token.compiled = Some(Box::new(compiled));

        Ok(AsmInfo {
            code_addr: 0,
            code_size: wasm_bytes.len(),
        })
    }

    fn compile_bridge(
        &mut self,
        _fail_descr: &dyn FailDescr,
        _inputargs: &[InputArg],
        _ops: &[Op],
        _original_token: &JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        Err(BackendError::Unsupported(
            "wasm backend: bridge compile not yet implemented".into(),
        ))
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("no compiled code")
            .downcast_ref::<CompiledWasmLoop>()
            .expect("not CompiledWasmLoop");

        // Allocate frame area large enough for slots + call trampoline area.
        // MIN_FRAME_BYTES accommodates the call area at offset 2000+.
        let min_slots = codegen::MIN_FRAME_BYTES / 8;
        let frame_size = min_slots.max(1 + compiled.max_output_slots.max(compiled.num_inputs));
        let mut frame = vec![0i64; frame_size];

        // Write inputs to frame[1..]
        for (i, arg) in args.iter().enumerate() {
            frame[1 + i] = match arg {
                Value::Int(v) => *v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.0 as i64,
                Value::Void => 0,
            };
        }

        // frame_ptr = byte offset of frame[0] in wasm linear memory
        let frame_ptr = frame.as_mut_ptr() as usize as u32;

        // Execute the compiled wasm function
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            js_glue::execute(compiled.func_handle, frame_ptr);
        }
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            panic!("wasm backend execute_token requires JS runtime");
        }

        // Read fail_index from frame[0]
        let fail_index = frame[0] as u32;
        let fail_descr = compiled
            .fail_descrs
            .get(fail_index as usize)
            .expect("invalid fail_index from compiled wasm");

        // Read output values
        let num_outputs = fail_descr.fail_arg_types.len();
        let raw_values: Vec<i64> = (0..num_outputs).map(|i| frame[1 + i]).collect();

        DeadFrame {
            data: Box::new(WasmFrameData {
                raw_values,
                fail_descr: fail_descr.clone(),
            }),
        }
    }

    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token(token, &values)
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        data.fail_descr.as_ref()
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        data.raw_values[index]
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        f64::from_bits(data.raw_values[index] as u64)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        GcRef(data.raw_values[index] as usize)
    }

    fn invalidate_loop(&self, _token: &JitCellToken) {
        // No native code to invalidate — wasm modules are immutable.
    }
}
