/// Compact binary encoding for traces.
///
/// Translates the concept from rpython/jit/metainterp/opencoder.py —
/// a compact binary format for serializing and deserializing traces.
///
/// Uses LEB128 variable-length integer encoding for compactness.
use majit_ir::{InputArg, Op, OpCode, OpRef, Type, OPCODE_COUNT};

use crate::trace::Trace;

/// Encode a u64 as a variable-length integer (LEB128).
pub fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

/// Decode a varint from a byte slice. Returns (value, bytes_consumed).
///
/// # Panics
///
/// Panics if the buffer is truncated (no terminating byte with high bit clear).
pub fn decode_varint(buf: &[u8]) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in buf.iter().enumerate() {
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return (value, i + 1);
        }
        shift += 7;
    }
    panic!("truncated varint");
}

fn type_to_u8(tp: Type) -> u8 {
    match tp {
        Type::Int => 0,
        Type::Ref => 1,
        Type::Float => 2,
        Type::Void => 3,
    }
}

fn u8_to_type(v: u8) -> Type {
    match v {
        0 => Type::Int,
        1 => Type::Ref,
        2 => Type::Float,
        3 => Type::Void,
        _ => panic!("invalid type byte: {v}"),
    }
}

fn u16_to_opcode(v: u16) -> OpCode {
    assert!(
        (v as usize) < OPCODE_COUNT,
        "invalid opcode discriminant: {v}"
    );
    // SAFETY: OpCode is #[repr(u16)] and we checked the discriminant is in range.
    unsafe { std::mem::transmute(v) }
}

/// Encode a Trace into a compact byte buffer.
pub fn encode_trace(trace: &Trace) -> Vec<u8> {
    let mut buf = Vec::new();

    // Encode input args count and types
    encode_varint(&mut buf, trace.inputargs.len() as u64);
    for arg in &trace.inputargs {
        buf.push(type_to_u8(arg.tp));
    }

    // Encode ops count
    encode_varint(&mut buf, trace.ops.len() as u64);

    // Encode each op
    for op in &trace.ops {
        encode_varint(&mut buf, op.opcode as u16 as u64);
        encode_varint(&mut buf, op.args.len() as u64);
        for arg in &op.args {
            encode_varint(&mut buf, arg.0 as u64);
        }
        // Encode whether there's a descriptor
        buf.push(if op.descr.is_some() { 1 } else { 0 });
    }

    buf
}

/// Decode a Trace from a compact byte buffer.
///
/// Note: descriptor references are not preserved — ops with descriptors
/// in the original trace will have `descr: None` after decoding, but
/// the has-descriptor flag is still decoded (and could be used to
/// reconstruct descriptors from a separate table).
pub fn decode_trace(buf: &[u8]) -> Trace {
    let mut pos = 0;

    // Decode input args
    let (num_inputargs, n) = decode_varint(&buf[pos..]);
    pos += n;
    let num_inputargs = num_inputargs as usize;

    let mut inputargs = Vec::with_capacity(num_inputargs);
    for i in 0..num_inputargs {
        let tp = u8_to_type(buf[pos]);
        pos += 1;
        inputargs.push(InputArg::from_type(tp, i as u32));
    }

    // Decode ops
    let (num_ops, n) = decode_varint(&buf[pos..]);
    pos += n;
    let num_ops = num_ops as usize;

    let mut ops = Vec::with_capacity(num_ops);
    for _ in 0..num_ops {
        let (opcode_raw, n) = decode_varint(&buf[pos..]);
        pos += n;
        let opcode = u16_to_opcode(opcode_raw as u16);

        let (num_args, n) = decode_varint(&buf[pos..]);
        pos += n;
        let num_args = num_args as usize;

        let mut args = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            let (arg_ref, n) = decode_varint(&buf[pos..]);
            pos += n;
            args.push(OpRef(arg_ref as u32));
        }

        let _has_descr = buf[pos];
        pos += 1;

        ops.push(Op::new(opcode, &args));
    }

    Trace::new(inputargs, ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        let values = [0u64, 1, 127, 128, 255, 256, 65535, 0xFFFF_FFFF, u64::MAX];
        for &val in &values {
            let mut buf = Vec::new();
            encode_varint(&mut buf, val);
            let (decoded, consumed) = decode_varint(&buf);
            assert_eq!(decoded, val, "roundtrip failed for {val}");
            assert_eq!(consumed, buf.len());
        }
    }

    #[test]
    fn test_varint_small_values() {
        // Values 0..=127 should encode to a single byte.
        for val in 0..=127u64 {
            let mut buf = Vec::new();
            encode_varint(&mut buf, val);
            assert_eq!(buf.len(), 1, "value {val} should be 1 byte");
            assert_eq!(buf[0], val as u8);
        }
    }

    #[test]
    fn test_varint_128_is_two_bytes() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 128);
        assert_eq!(buf.len(), 2);
        let (decoded, consumed) = decode_varint(&buf);
        assert_eq!(decoded, 128);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_empty_trace_roundtrip() {
        let trace = Trace::new(vec![], vec![]);
        let encoded = encode_trace(&trace);
        let decoded = decode_trace(&encoded);
        assert_eq!(decoded.num_inputargs(), 0);
        assert_eq!(decoded.num_ops(), 0);
    }

    #[test]
    fn test_trace_roundtrip() {
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(2)]),
        ];
        let trace = Trace::new(inputargs, ops);

        let encoded = encode_trace(&trace);
        let decoded = decode_trace(&encoded);

        assert_eq!(decoded.num_inputargs(), 3);
        assert_eq!(decoded.inputargs[0].tp, Type::Int);
        assert_eq!(decoded.inputargs[0].index, 0);
        assert_eq!(decoded.inputargs[1].tp, Type::Ref);
        assert_eq!(decoded.inputargs[1].index, 1);
        assert_eq!(decoded.inputargs[2].tp, Type::Float);
        assert_eq!(decoded.inputargs[2].index, 2);

        assert_eq!(decoded.num_ops(), 3);
        assert_eq!(decoded.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(decoded.ops[0].args.as_slice(), &[OpRef(0), OpRef(0)]);
        assert_eq!(decoded.ops[1].opcode, OpCode::GuardTrue);
        assert_eq!(decoded.ops[1].args.as_slice(), &[OpRef(1)]);
        assert_eq!(decoded.ops[2].opcode, OpCode::Jump);
        assert_eq!(decoded.ops[2].args.as_slice(), &[OpRef(2)]);

        assert!(decoded.is_loop());
    }

    #[test]
    fn test_trace_with_many_ops() {
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();

        // Build a chain of IntAdd ops
        for i in 0..100u32 {
            ops.push(Op::new(OpCode::IntAdd, &[OpRef(i), OpRef(0)]));
        }
        ops.push(Op::new(OpCode::Jump, &[OpRef(100)]));

        let trace = Trace::new(inputargs, ops);
        let encoded = encode_trace(&trace);
        let decoded = decode_trace(&encoded);

        assert_eq!(decoded.num_ops(), 101);
        for i in 0..100 {
            assert_eq!(decoded.ops[i].opcode, OpCode::IntAdd);
            assert_eq!(decoded.ops[i].args[0], OpRef(i as u32));
            assert_eq!(decoded.ops[i].args[1], OpRef(0));
        }
        assert_eq!(decoded.ops[100].opcode, OpCode::Jump);
        assert!(decoded.is_loop());
    }

    #[test]
    fn test_trace_with_finish() {
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::new(OpCode::Finish, &[OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);
        let encoded = encode_trace(&trace);
        let decoded = decode_trace(&encoded);

        assert!(decoded.is_finished());
        assert!(!decoded.is_loop());
    }

    #[test]
    fn test_trace_preserves_descr_flag() {
        // Verify that the has-descr byte is encoded (even though we don't
        // reconstruct the descriptor on decode).
        let ops = vec![Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)])];
        let trace = Trace::new(vec![InputArg::new_int(0)], ops);

        let encoded = encode_trace(&trace);
        // The last byte of the encoded op should be the descr flag (0 = no descr)
        assert_eq!(*encoded.last().unwrap(), 0);
    }

    #[test]
    fn test_varint_multiple_in_buffer() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 42);
        encode_varint(&mut buf, 12345);
        encode_varint(&mut buf, 0);

        let (v1, n1) = decode_varint(&buf);
        assert_eq!(v1, 42);
        let (v2, n2) = decode_varint(&buf[n1..]);
        assert_eq!(v2, 12345);
        let (v3, _n3) = decode_varint(&buf[n1 + n2..]);
        assert_eq!(v3, 0);
    }
}
