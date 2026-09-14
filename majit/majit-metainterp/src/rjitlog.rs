//! `rpython/rlib/rjitlog/rjitlog.py`.
//!
//! Binary jitlog writer. `JITLOG` names the output file
//! (`rjitlog.c jitlog_try_init_using_env`). The trace id increments even
//! when the file is unset.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;

use majit_ir::operand::Operand;
use majit_ir::{InputArg, Op, OpCode, Type, Value};

/// `rjitlog.py` mark table, `start = 0x11`.
const MARK_BASE: u8 = 0x11;
pub const MARK_INPUT_ARGS: u8 = MARK_BASE;
pub const MARK_RESOP_META: u8 = MARK_BASE + 1;
pub const MARK_RESOP: u8 = MARK_BASE + 2;
pub const MARK_RESOP_DESCR: u8 = MARK_BASE + 3;
pub const MARK_TRACE: u8 = MARK_BASE + 6;
pub const MARK_TRACE_OPT: u8 = MARK_BASE + 7;
pub const MARK_START_TRACE: u8 = MARK_BASE + 10;
pub const MARK_JITLOG_HEADER: u8 = MARK_BASE + 13;
pub const MARK_MERGE_POINT: u8 = MARK_BASE + 14;
pub const MARK_ABORT_TRACE: u8 = MARK_BASE + 16;

/// `rjitlog.py JITLOG_VERSION`.
const JITLOG_VERSION: u16 = 4;

struct JitLogState {
    file: Option<File>,
    trace_id: u64,
}

static JITLOG: Mutex<JitLogState> = Mutex::new(JitLogState {
    file: None,
    trace_id: 0,
});

fn lock() -> std::sync::MutexGuard<'static, JitLogState> {
    JITLOG.lock().unwrap_or_else(|e| e.into_inner())
}

/// `rjitlog.c jitlog_try_init_using_env`.
pub fn jitlog_try_init_using_env() {
    let Some(path) = std::env::var_os("JITLOG") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let mut state = lock();
    if state.file.is_some() {
        return;
    }
    match File::create(&path) {
        Ok(file) => {
            state.file = Some(file);
            // rjitlog.py JitLogger.setup_once: header before any trace.
            let header = assemble_header();
            write_marked(&mut state, MARK_JITLOG_HEADER, &header);
        }
        Err(err) => eprintln!("could not open '{}': {err}", path.to_string_lossy()),
    }
}

/// `rjitlog.py assemble_header`.
fn assemble_header() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&JITLOG_VERSION.to_le_bytes());
    content.push(0); // 64-bit
    content.extend_from_slice(&encode_str(&std::env::consts::ARCH));
    content.push(MARK_RESOP_META);
    let opcodes: Vec<OpCode> = OpCode::all().collect();
    content.extend_from_slice(&encode_le_16bit(opcodes.len() as u16));
    for op in opcodes {
        content.extend_from_slice(&encode_le_16bit(op.as_u16()));
        content.extend_from_slice(&encode_str(&op.name().to_ascii_lowercase()));
    }
    content
}

/// `rjitlog.c jitlog_enabled`.
pub fn jitlog_enabled() -> bool {
    lock().file.is_some()
}

/// `rjitlog.py JitLogger.start_new_trace`. Increments even when disabled.
///
/// `descr_or_entry` is `compute_unique_id(faildescr)` for a bridge, or
/// `int(entry_bridge)` for a loop.
pub fn start_new_trace(is_bridge: bool, descr_or_entry: u64, jd_name: &str) -> u64 {
    jitlog_try_init_using_env();
    let mut state = lock();
    state.trace_id += 1;
    let tid = state.trace_id;
    if state.file.is_some() {
        let kind = if is_bridge { "bridge" } else { "loop" };
        let mut payload = encode_le_addr(tid).to_vec();
        payload.extend_from_slice(&encode_str(kind));
        payload.extend_from_slice(&encode_le_addr(descr_or_entry));
        payload.extend_from_slice(&encode_str(jd_name));
        write_marked(&mut state, MARK_START_TRACE, &payload);
    }
    tid
}

/// `rjitlog.py JitLogger.trace_aborted`.
pub fn trace_aborted(tid: u64) {
    let mut state = lock();
    if state.file.is_none() {
        return;
    }
    let payload = encode_le_addr(tid);
    write_marked(&mut state, MARK_ABORT_TRACE, &payload);
}

/// Current `JitLogger.trace_id`.
pub fn trace_id() -> u64 {
    lock().trace_id
}

/// `rjitlog.py JitLogger.log_trace` + `LogTrace.write`.
///
/// `tid` is the id `start_new_trace` returned for this compile, not the
/// process-global latest id (`JitLogger.trace_id` on the logger that
/// started the write).
pub fn write_trace<A, O>(tag: u8, tid: u64, inputargs: &[A], ops: &[O])
where
    A: Borrow<InputArg>,
    O: Borrow<Op>,
{
    jitlog_try_init_using_env();
    let mut state = lock();
    if state.file.is_none() {
        return;
    }
    let header = encode_le_addr(tid);
    write_marked(&mut state, tag, &header);
    let mut memo = VarMemo::default();
    let args: Vec<String> = inputargs
        .iter()
        .map(|arg| memo.inputarg(arg.borrow()))
        .collect();
    write_marked(&mut state, MARK_INPUT_ARGS, &encode_str(&args.join(",")));
    for op in ops {
        let op = op.borrow();
        if op.opcode == OpCode::DebugMergePoint {
            // rjitlog.py LogTrace.encode_debug_info: skip when
            // get_location is unset. Do not emit MARK_MERGE_POINT
            // without MARK_INIT_MERGE_POINT.
            continue;
        }
        write_resop(&mut state, &mut memo, op);
    }
}

/// `rjitlog.py LogTrace.var_to_str` memo.
#[derive(Default)]
struct VarMemo {
    next: usize,
    ids: HashMap<(u8, u64), usize>,
}

impl VarMemo {
    fn assign(&mut self, kind: u8, key: u64) -> usize {
        *self.ids.entry((kind, key)).or_insert_with(|| {
            let id = self.next;
            self.next += 1;
            id
        })
    }

    fn inputarg(&mut self, arg: &InputArg) -> String {
        let kind = match arg.tp {
            Type::Int => b'i',
            Type::Ref => b'p',
            Type::Float => b'f',
            Type::Void => b'?',
        };
        let id = self.assign(kind, arg.index as u64);
        format!("{}{id}", kind as char)
    }

    fn op_result(&mut self, op: &Op) -> String {
        // rjitlog.py var_to_str: void / unknown type is `?`, and the
        // memo slot is still allocated.
        let kind = match op.type_ {
            Type::Int => b'i',
            Type::Ref => b'p',
            Type::Float => b'f',
            Type::Void => b'?',
        };
        let id = self.assign(kind, op.pos().get().raw() as u64);
        if op.type_ == Type::Void {
            return "?".into();
        }
        format!("{}{id}", kind as char)
    }

    /// `rjitlog.py LogTrace.var_to_str`.
    fn operand(&mut self, arg: &Operand) -> String {
        if arg.is_none() {
            return "-".into();
        }
        if let Some(value) = arg.const_value() {
            // rjitlog.py var_to_str: ConstInt / ConstFloat / ConstPtr
            // allocate a memo slot before formatting. Value::Void is
            // not an upstream constant; do not consume a slot.
            return match value {
                Value::Int(v) => {
                    let _ = self.assign(b'I', v as u64);
                    v.to_string()
                }
                Value::Float(v) => {
                    let _ = self.assign(b'F', v.to_bits());
                    v.to_string()
                }
                Value::Ref(r) if r.is_null() => {
                    let _ = self.assign(b'P', 0);
                    "ConstPtr(null)".into()
                }
                Value::Ref(r) => {
                    let id = self.assign(b'P', r.0 as u64);
                    format!("ConstPtr(ptr{id})")
                }
                Value::Void => "None".into(),
            };
        }
        if arg.is_null_ref() {
            let _ = self.assign(b'P', 0);
            return "ConstPtr(null)".into();
        }
        if arg.is_inputarg() {
            let opref = arg.to_opref();
            let tp = arg.type_();
            let idx = opref.raw() as u64;
            let kind = match tp {
                Type::Int => b'i',
                Type::Ref => b'p',
                Type::Float => b'f',
                Type::Void => b'?',
            };
            let id = self.assign(kind, idx);
            return format!("{}{id}", kind as char);
        }
        if arg.is_resop() {
            let kind = match arg.type_() {
                Type::Int => b'i',
                Type::Ref => b'p',
                Type::Float => b'f',
                Type::Void => b'v',
            };
            let id = self.assign(kind, arg.to_opref().raw() as u64);
            return format!("{}{id}", kind as char);
        }
        "?".into()
    }
}

/// `rjitlog.py LogTrace.encode_op`.
fn write_resop(state: &mut JitLogState, memo: &mut VarMemo, op: &Op) {
    let descr = op.getdescr();
    let mark = if descr.is_some() {
        MARK_RESOP_DESCR
    } else {
        MARK_RESOP
    };
    let mut line = encode_le_16bit(op.opcode.as_u16()).to_vec();
    // rjitlog.py encode_op: memoize arguments, then the result.
    let arg_strs: Vec<String> = op
        .getarglist()
        .iter()
        .map(|arg| memo.operand(arg))
        .collect();
    let mut body = memo.op_result(op);
    for arg in arg_strs {
        body.push(',');
        body.push_str(&arg);
    }
    if let Some(ref d) = descr {
        // rjitlog.py encode_op: descr.repr_of_descr().
        body.push(',');
        body.push_str(&d.repr());
    }
    line.extend_from_slice(&encode_str(&body));
    if let Some(d) = descr {
        line.extend_from_slice(&encode_le_addr(
            std::sync::Arc::as_ptr(&d) as *const () as u64
        ));
    }
    let failargs = match op.getfailargs() {
        Some(args) => args
            .iter()
            .map(|arg| memo.operand(arg))
            .collect::<Vec<_>>()
            .join(","),
        None => String::new(),
    };
    line.extend_from_slice(&encode_str(&failargs));
    write_marked(state, mark, &line);
}

fn write_marked(state: &mut JitLogState, mark: u8, payload: &[u8]) {
    let Some(file) = state.file.as_mut() else {
        return;
    };
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(mark);
    buf.extend_from_slice(payload);
    let _ = file.write_all(&buf);
}

/// `rjitlog.py encode_le_16bit`.
pub fn encode_le_16bit(val: u16) -> [u8; 2] {
    val.to_le_bytes()
}

/// `rjitlog.py encode_le_addr` on 64-bit.
pub fn encode_le_addr(val: u64) -> [u8; 8] {
    val.to_le_bytes()
}

/// `rjitlog.py encode_str`.
pub fn encode_str(string: &str) -> Vec<u8> {
    let len = string.len() as u32;
    let mut out = Vec::with_capacity(4 + string.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(string.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_trace_is_the_seventh_mark() {
        assert_eq!(MARK_TRACE, 0x17);
        assert_eq!(MARK_START_TRACE, 0x1b);
        assert_eq!(MARK_JITLOG_HEADER, 0x1e);
        assert_eq!(MARK_MERGE_POINT, 0x1f);
        assert_eq!(MARK_ABORT_TRACE, 0x21);
    }

    #[test]
    fn encode_str_prefixes_little_endian_length() {
        let encoded = encode_str("ab");
        assert_eq!(&encoded[..4], &[2, 0, 0, 0]);
        assert_eq!(&encoded[4..], b"ab");
    }

    #[test]
    fn start_new_trace_increments_when_disabled() {
        let before = trace_id();
        let tid = start_new_trace(false, 0, "test");
        assert!(tid > before);
    }

    #[test]
    fn write_trace_is_a_no_op_when_disabled() {
        write_trace::<InputArg, Op>(MARK_TRACE, 0, &[], &[]);
        write_trace::<InputArg, Op>(MARK_TRACE_OPT, 0, &[], &[]);
    }

    #[test]
    fn var_to_str_names_inputargs_i0_p1() {
        let mut memo = VarMemo::default();
        let i = InputArg::new_int(0);
        let p = InputArg::from_type(Type::Ref, 1);
        assert_eq!(memo.inputarg(&i), "i0");
        assert_eq!(memo.inputarg(&p), "p1");
        assert_eq!(memo.inputarg(&i), "i0");
    }

    #[test]
    fn void_result_is_question_mark() {
        let mut memo = VarMemo::default();
        let op = Op::new(OpCode::Finish, &[]);
        assert_eq!(memo.op_result(&op), "?");
    }

    #[test]
    fn const_operands_reserve_a_memo_slot() {
        let mut memo = VarMemo::default();
        assert_eq!(memo.operand(&Operand::none()), "-");
        assert_eq!(memo.operand(&Operand::const_from_value(Value::Int(5))), "5");
        assert_eq!(
            memo.operand(&Operand::const_from_value(Value::Ref(majit_ir::GcRef(
                0x1000
            )))),
            "ConstPtr(ptr1)"
        );
        // ConstInt reserved slot 0; ConstPtr reserved slot 1; the next
        // named box is 2. Value::Void must not steal a slot.
        assert_eq!(
            memo.operand(&Operand::const_from_value(Value::Void)),
            "None"
        );
        let i = InputArg::new_int(0);
        assert_eq!(memo.inputarg(&i), "i2");
    }
}
