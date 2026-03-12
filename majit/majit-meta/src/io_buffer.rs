use std::cell::RefCell;
use std::fmt;
use std::io::{self, Write};

use majit_ir::{OpCode, Type};

use crate::call_descr::make_call_descr;
use crate::trace_ctx::TraceCtx;

// ── Thread-local I/O buffer ──────────────────────────────────────────
//
// JIT-compiled loops call extern "C" shim functions that write to this
// buffer instead of directly to stdout. At each successful loop iteration,
// commit flushes the buffer. On guard failure, discard clears it.

thread_local! {
    static JIT_IO_BUFFER: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(4096));
}

/// Write raw bytes to the JIT I/O buffer.
pub fn io_buffer_write(data: &[u8]) {
    JIT_IO_BUFFER.with(|buf| {
        buf.borrow_mut().extend_from_slice(data);
    });
}

/// Write formatted output to the JIT I/O buffer.
pub fn io_buffer_write_fmt(args: fmt::Arguments<'_>) {
    JIT_IO_BUFFER.with(|buf| {
        let _ = buf.borrow_mut().write_fmt(args);
    });
}

/// Flush the JIT I/O buffer to stdout.
pub fn io_buffer_commit() {
    JIT_IO_BUFFER.with(|buf| {
        let mut b = buf.borrow_mut();
        if !b.is_empty() {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            let _ = out.write_all(&b);
            b.clear();
        }
    });
}

/// Discard the JIT I/O buffer contents.
pub fn io_buffer_discard() {
    JIT_IO_BUFFER.with(|buf| {
        buf.borrow_mut().clear();
    });
}

// The actual extern "C" function that commit_io CallN calls.
extern "C" fn jit_commit_io_shim() {
    io_buffer_commit();
}

/// Emit a CallN to commit the I/O buffer in the trace.
///
/// Should be called right before returning `TraceAction::CloseLoop`,
/// so that each successful loop iteration flushes its I/O.
pub fn emit_commit_io(ctx: &mut TraceCtx) {
    let func_ref = ctx.const_int(jit_commit_io_shim as *const () as usize as i64);
    let descr = make_call_descr(&[], Type::Void);
    ctx.record_op_with_descr(OpCode::CallN, &[func_ref], descr);
}
