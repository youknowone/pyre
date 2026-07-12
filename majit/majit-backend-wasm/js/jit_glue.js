// JIT glue layer for majit-backend-wasm.
//
// Called from Rust (via wasm-bindgen) to compile and execute
// dynamically generated wasm modules inside the browser.

let funcTable = {};
let nextFuncId = 1;
let mainMemory = null;
let mainTable = null;

// Call area offsets (must match codegen.rs constants).
const CALL_RESULT_OFS = 2000;
const CALL_FUNC_OFS = 2008;
const CALL_NARGS_OFS = 2016;
const CALL_ARGS_OFS = 2024;

export function jit_set_memory(mem) {
  mainMemory = mem;
}

export function jit_set_table(table) {
  mainTable = table;
}

// Call trampoline — invoked by generated wasm when it encounters a CALL op.
// Reads func_ptr + args from the frame's call area, performs the call via
// the main module's indirect function table, writes result back.
function jitCallTrampoline(framePtr, callAreaOfs = CALL_RESULT_OFS) {
  const view = new DataView(mainMemory.buffer);
  const funcPtrLo = view.getUint32(framePtr + callAreaOfs + 8, true);
  const numArgs = Number(view.getBigInt64(framePtr + callAreaOfs + 16, true));

  // Read args from call area
  const args = [];
  for (let i = 0; i < numArgs; i++) {
    // On wasm32, values are i32 (pointers/ints). Read low 32 bits of each i64 slot.
    args.push(view.getInt32(framePtr + callAreaOfs + 24 + i * 8, true));
  }

  // Call via the main module's function table
  let result = 0;
  if (mainTable) {
    try {
      const fn = mainTable.get(funcPtrLo);
      result = fn(...args) ?? 0;
    } catch (e) {
      console.error('[jit_call] trampoline error:', e);
      result = 0;
    }
  }

  // Write result to call area (as i64: low 32 bits = result, high 32 bits = 0)
  view.setInt32(framePtr + callAreaOfs, result, true);
  view.setInt32(framePtr + callAreaOfs + 4, 0, true);
}

export function jit_compile_wasm(bytesPtr, bytesLen) {
  if (!mainMemory) {
    throw new Error("jit_set_memory() must be called before jit_compile_wasm()");
  }
  const bytes = new Uint8Array(mainMemory.buffer, bytesPtr, bytesLen).slice();
  const module = new WebAssembly.Module(bytes);
  const imports = { env: { memory: mainMemory } };

  // Check if the module needs jit_call import
  // (wasm-encoder adds it when trace has CALL ops)
  try {
    // `__indirect_function_table` is reserved for inter-trace call_indirect
    // chaining; the module imports it only when it has CALL ops. Extra
    // entries in the import object are ignored when not declared.
    const instance = new WebAssembly.Instance(module, {
      env: { memory: mainMemory, jit_call: jitCallTrampoline, jit_call_compact: jitCallTrampoline, __indirect_function_table: mainTable }
    });
    return registerTrace(instance.exports.trace);
  } catch (e) {
    // Retry without jit_call (for traces without CALL ops)
    const instance = new WebAssembly.Instance(module, {
      env: { memory: mainMemory }
    });
    return registerTrace(instance.exports.trace);
  }
}

// Append a compiled trace to the shared indirect function table and use its
// table slot as the id, mirroring the wasmtime host. The slot is both the
// jit_execute_wasm handle and the index an in-module call_indirect targets.
// Falls back to a private counter when no table is available.
function registerTrace(traceFn) {
  let id;
  if (mainTable) {
    id = mainTable.grow(1);
    mainTable.set(id, traceFn);
  } else {
    id = nextFuncId++;
  }
  funcTable[id] = traceFn;
  return id;
}

export function jit_execute_wasm(funcId, framePtr) {
  const func = funcTable[funcId];
  if (!func) {
    throw new Error(`jit_execute_wasm: unknown funcId ${funcId}`);
  }
  return func(framePtr);
}

export function jit_free_wasm(funcId) {
  delete funcTable[funcId];
}
