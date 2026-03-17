/// JIT-enabled tiny3 interpreter using JitDriver + JitState.
///
/// Greens: [pos, bytecode]   (bytecode is constant per trace — not tracked)
/// Reds:   [args, arg_types]
///
/// Type promotion strategy matching RPython's tiny3_hotpath.py `promote(y.__class__)`:
/// Each arg carries a ValueType tag (Int=0, Float=1). At arithmetic operations,
/// we emit GuardValue on both type tags to specialize the trace for a specific
/// type combination. Int+Int traces to IntAdd/IntSub/IntMul, otherwise
/// FloatAdd/FloatSub/FloatMul with f64 bit patterns stored as i64.
///
/// This example hand-writes `trace_instruction` for educational purposes.
/// In production, the `#[jit_interp]` proc macro auto-generates tracing
/// code from the interpreter's match dispatch — see aheuijit for an example.
use majit_ir::{OpCode, OpRef, Type};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

const DEFAULT_THRESHOLD: u32 = 3;

// ── Value types ──

/// Type tag for each arg slot. Mirrors IntBox/FloatBox class identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueType {
    Int = 0,
    Float = 1,
}

impl ValueType {
    fn from_tag(tag: i64) -> Self {
        match tag {
            0 => ValueType::Int,
            1 => ValueType::Float,
            _ => panic!("unknown type tag: {tag}"),
        }
    }

    fn tag(self) -> i64 {
        self as i64
    }
}

// ── JitState types ──

/// Red variables: args array + per-arg type tags.
/// Float values stored as f64 bits cast to i64.
pub struct Tiny3State {
    args: Vec<i64>,
    arg_types: Vec<ValueType>,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct Tiny3Meta {
    num_args: usize,
    /// Type snapshot at trace start — available for type compatibility checks.
    #[allow(dead_code)]
    arg_types: Vec<ValueType>,
}

/// Symbolic state during tracing — OpRef for each arg + type tag + computation stack.
pub struct Tiny3Sym {
    /// args[i] → current OpRef mapping.
    trace_args: Vec<OpRef>,
    /// args[i] → type tag OpRef mapping.
    trace_type_args: Vec<OpRef>,
    /// Intermediate computation stack during tracing (value, type_ref).
    trace_stack: Vec<(OpRef, OpRef)>,
}

impl JitState for Tiny3State {
    type Meta = Tiny3Meta;
    type Sym = Tiny3Sym;
    type Env = [&'static str];

    fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Tiny3Meta {
        Tiny3Meta {
            num_args: self.args.len(),
            arg_types: self.arg_types.clone(),
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        // Interleave: [val0, type0, val1, type1, ...]
        let mut live = Vec::with_capacity(self.args.len() * 2);
        for i in 0..self.args.len() {
            live.push(self.args[i]);
            live.push(self.arg_types[i].tag());
        }
        live
    }

    fn live_value_types(&self, _meta: &Self::Meta) -> Vec<Type> {
        let mut types = Vec::with_capacity(self.args.len() * 2);
        for t in &self.arg_types {
            match t {
                ValueType::Int => types.push(Type::Int),
                ValueType::Float => types.push(Type::Float),
            }
            // Type tag is always Int
            types.push(Type::Int);
        }
        types
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> Tiny3Sym {
        // Input args are interleaved: [val0, type0, val1, type1, ...]
        let num = meta.num_args;
        let trace_args: Vec<OpRef> = (0..num).map(|i| OpRef((i * 2) as u32)).collect();
        let trace_type_args: Vec<OpRef> = (0..num).map(|i| OpRef((i * 2 + 1) as u32)).collect();
        Tiny3Sym {
            trace_args,
            trace_type_args,
            trace_stack: Vec::new(),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        self.args.len() == meta.num_args
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        // Deinterleave: [val0, type0, val1, type1, ...]
        let num = values.len() / 2;
        self.args.clear();
        self.arg_types.clear();
        for i in 0..num {
            self.args.push(values[i * 2]);
            self.arg_types.push(ValueType::from_tag(values[i * 2 + 1]));
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        // Interleave: [val0, type0, val1, type1, ...]
        let mut args = Vec::with_capacity(sym.trace_args.len() * 2);
        for i in 0..sym.trace_args.len() {
            args.push(sym.trace_args[i]);
            args.push(sym.trace_type_args[i]);
        }
        args
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

/// Trace one instruction, recording IR into ctx.
/// `stack_types` provides the runtime types of values on the computation stack.
fn trace_instruction(
    ctx: &mut TraceCtx,
    sym: &mut Tiny3Sym,
    stack_types: &[ValueType],
    bytecode: &[&str],
    pos: usize,
) -> TraceAction {
    let opcode = bytecode[pos];

    if opcode == "ADD" || opcode == "SUB" || opcode == "MUL" {
        let (b_ref, b_type_ref) = sym.trace_stack.pop().unwrap();
        let (a_ref, a_type_ref) = sym.trace_stack.pop().unwrap();

        // Promote type tags: GuardValue on each type to make it a compile-time constant.
        // This mirrors RPython's promote(y.__class__) / promote(x.__class__).
        let num_live = sym.trace_args.len() * 2;
        let fail_args: Vec<OpRef> = collect_fail_args(sym);

        // Read runtime types from the concrete stack.
        let stack_len = stack_types.len();
        let a_type = stack_types[stack_len - 2];
        let b_type = stack_types[stack_len - 1];

        let a_type_const = ctx.const_int(a_type.tag());
        ctx.record_guard_with_fail_args(
            OpCode::GuardValue,
            &[a_type_ref, a_type_const],
            num_live,
            &fail_args,
        );
        let b_type_const = ctx.const_int(b_type.tag());
        ctx.record_guard_with_fail_args(
            OpCode::GuardValue,
            &[b_type_ref, b_type_const],
            num_live,
            &fail_args,
        );

        if a_type == ValueType::Int && b_type == ValueType::Int {
            let ir_op = match opcode {
                "ADD" => OpCode::IntAdd,
                "SUB" => OpCode::IntSub,
                "MUL" => OpCode::IntMul,
                _ => unreachable!(),
            };
            let result = ctx.record_op(ir_op, &[a_ref, b_ref]);
            let int_type = ctx.const_int(ValueType::Int.tag());
            sym.trace_stack.push((result, int_type));
        } else {
            // At least one operand is float — cast ints to float, do float op.
            let fa = if a_type == ValueType::Int {
                ctx.record_op(OpCode::CastIntToFloat, &[a_ref])
            } else {
                a_ref
            };
            let fb = if b_type == ValueType::Int {
                ctx.record_op(OpCode::CastIntToFloat, &[b_ref])
            } else {
                b_ref
            };
            let ir_op = match opcode {
                "ADD" => OpCode::FloatAdd,
                "SUB" => OpCode::FloatSub,
                "MUL" => OpCode::FloatMul,
                _ => unreachable!(),
            };
            let result = ctx.record_op(ir_op, &[fa, fb]);
            let float_type = ctx.const_int(ValueType::Float.tag());
            sym.trace_stack.push((result, float_type));
        }
    } else if opcode.starts_with('#') {
        let n = parse_int(opcode, 1) as usize;
        let opref = sym.trace_args[n - 1];
        let type_ref = sym.trace_type_args[n - 1];
        sym.trace_stack.push((opref, type_ref));
    } else if opcode.starts_with("->#") {
        let n = parse_int(opcode, 3) as usize;
        let (opref, type_ref) = sym.trace_stack.pop().unwrap();
        sym.trace_args[n - 1] = opref;
        sym.trace_type_args[n - 1] = type_ref;
    } else if opcode == "{" {
        return TraceAction::Abort;
    } else if opcode == "}" {
        let (flag_ref, _flag_type) = sym.trace_stack.pop().unwrap();
        let zero = ctx.const_int(0);
        let cond = ctx.record_op(OpCode::IntNe, &[flag_ref, zero]);
        let num_live = sym.trace_args.len() * 2;
        let fail_args = collect_fail_args(sym);
        ctx.record_guard_with_fail_args(OpCode::GuardTrue, &[cond], num_live, &fail_args);
        return TraceAction::CloseLoop;
    } else if let Some(fval) = try_parse_float(opcode) {
        // Float literal
        let bits = fval.to_bits() as i64;
        let opref = ctx.const_int(bits);
        let float_type = ctx.const_int(ValueType::Float.tag());
        sym.trace_stack.push((opref, float_type));
    } else {
        // Integer literal
        let val = parse_int(opcode, 0);
        let opref = ctx.const_int(val);
        let int_type = ctx.const_int(ValueType::Int.tag());
        sym.trace_stack.push((opref, int_type));
    }

    TraceAction::Continue
}

/// Build interleaved fail_args from symbolic state: [val0, type0, val1, type1, ...]
fn collect_fail_args(sym: &Tiny3Sym) -> Vec<OpRef> {
    let mut fa = Vec::with_capacity(sym.trace_args.len() * 2);
    for i in 0..sym.trace_args.len() {
        fa.push(sym.trace_args[i]);
        fa.push(sym.trace_type_args[i]);
    }
    fa
}

pub struct JitTiny3Interp {
    driver: JitDriver<Tiny3State>,
}

impl JitTiny3Interp {
    pub fn new() -> Self {
        JitTiny3Interp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    /// Run a word-based program with integer args.
    /// Returns the result: stack top if non-empty, else args[0].
    pub fn run(&mut self, bytecode: &[&str], args: &mut Vec<i64>) -> i64 {
        let mut box_args: Vec<crate::interp::Box> =
            args.iter().map(|&v| crate::interp::Box::Int(v)).collect();
        let result = self.run_typed(bytecode, &mut box_args);
        // Sync back: integer args come back as integers
        args.clear();
        for b in &box_args {
            args.push(b.as_int());
        }
        result.as_int()
    }

    /// Run a word-based program with typed (Int/Float) args.
    /// Returns the result as a Box.
    pub fn run_typed(
        &mut self,
        bytecode: &[&str],
        args: &mut Vec<crate::interp::Box>,
    ) -> crate::interp::Box {
        // Safety: bytecode references are valid for the duration of run().
        let static_bytecode: &[&'static str] =
            unsafe { std::mem::transmute::<&[&str], &[&'static str]>(bytecode) };

        let (raw_args, arg_types) = boxes_to_raw(args);
        let mut state = Tiny3State {
            args: raw_args,
            arg_types,
        };
        let mut stack: Vec<crate::interp::Box> = Vec::new();
        let mut loops: Vec<usize> = Vec::new();
        let mut pos: usize = 0;

        while pos < bytecode.len() {
            // Capture stack types for tracing before merge_point.
            let stack_types: Vec<ValueType> = stack.iter().map(box_to_value_type).collect();

            // jit_merge_point
            {
                let bc = bytecode;
                let p = pos;
                self.driver.merge_point(|ctx, sym| {
                    trace_instruction(ctx, sym, &stack_types, bc, p)
                });
            }

            let opcode = bytecode[pos];
            pos += 1;

            if opcode == "ADD" || opcode == "SUB" || opcode == "MUL" {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let result = match (&a, &b) {
                    (crate::interp::Box::Int(ia), crate::interp::Box::Int(ib)) => {
                        let r = match opcode {
                            "ADD" => ia + ib,
                            "SUB" => ia - ib,
                            "MUL" => ia * ib,
                            _ => unreachable!(),
                        };
                        crate::interp::Box::Int(r)
                    }
                    _ => {
                        let fa = a.as_float();
                        let fb = b.as_float();
                        let r = match opcode {
                            "ADD" => fa + fb,
                            "SUB" => fa - fb,
                            "MUL" => fa * fb,
                            _ => unreachable!(),
                        };
                        crate::interp::Box::Float(r)
                    }
                };
                stack.push(result);
            } else if opcode.starts_with('#') {
                let n = parse_int(opcode, 1) as usize;
                let val = raw_to_box(state.args[n - 1], state.arg_types[n - 1]);
                stack.push(val);
            } else if opcode.starts_with("->#") {
                let n = parse_int(opcode, 3) as usize;
                let val = stack.pop().unwrap();
                let (raw, vt) = box_to_raw(&val);
                state.args[n - 1] = raw;
                state.arg_types[n - 1] = vt;
            } else if opcode == "{" {
                loops.push(pos);
            } else if opcode == "}" {
                let flag = stack.pop().unwrap();
                if flag.as_int() == 0 {
                    loops.pop();
                } else {
                    let target = *loops.last().unwrap();

                    // can_enter_jit
                    if target < pos
                        && self
                            .driver
                            .back_edge(target, &mut state, static_bytecode, || {})
                    {
                        // Compiled code ran — state is restored.
                        // The guard exits when flag == 0, so the loop is done.
                        loops.pop();
                        sync_state_to_boxes(&state, args);
                        continue;
                    }

                    pos = target;
                }
            } else if let Some(fval) = try_parse_float(opcode) {
                stack.push(crate::interp::Box::Float(fval));
            } else {
                // Integer literal
                stack.push(crate::interp::Box::Int(parse_int(opcode, 0)));
            }
        }

        // Sync final state back
        sync_state_to_boxes(&state, args);

        if !stack.is_empty() {
            stack.pop().unwrap()
        } else {
            raw_to_box(state.args[0], state.arg_types[0])
        }
    }
}

// ── Conversion helpers ──

fn box_to_value_type(b: &crate::interp::Box) -> ValueType {
    match b {
        crate::interp::Box::Int(_) => ValueType::Int,
        crate::interp::Box::Float(_) => ValueType::Float,
    }
}

fn box_to_raw(b: &crate::interp::Box) -> (i64, ValueType) {
    match b {
        crate::interp::Box::Int(v) => (*v, ValueType::Int),
        crate::interp::Box::Float(v) => (v.to_bits() as i64, ValueType::Float),
    }
}

fn raw_to_box(raw: i64, vt: ValueType) -> crate::interp::Box {
    match vt {
        ValueType::Int => crate::interp::Box::Int(raw),
        ValueType::Float => crate::interp::Box::Float(f64::from_bits(raw as u64)),
    }
}

fn boxes_to_raw(boxes: &[crate::interp::Box]) -> (Vec<i64>, Vec<ValueType>) {
    let mut vals = Vec::with_capacity(boxes.len());
    let mut types = Vec::with_capacity(boxes.len());
    for b in boxes {
        let (v, t) = box_to_raw(b);
        vals.push(v);
        types.push(t);
    }
    (vals, types)
}

fn sync_state_to_boxes(state: &Tiny3State, args: &mut Vec<crate::interp::Box>) {
    args.clear();
    for i in 0..state.args.len() {
        args.push(raw_to_box(state.args[i], state.arg_types[i]));
    }
}

/// Try parsing a float literal (must contain '.').
fn try_parse_float(s: &str) -> Option<f64> {
    if s.contains('.') {
        s.parse::<f64>().ok()
    } else {
        None
    }
}

fn parse_int(s: &str, start: usize) -> i64 {
    let s = &s[start..];
    let mut res: i64 = 0;
    for c in s.chars() {
        let d = c as i64 - '0' as i64;
        res = res * 10 + d;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    #[test]
    fn jit_fibonacci_single() {
        let prog: Vec<&str> = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![1i64, 1, 11];
        let result = jit.run(&prog, &mut args);
        assert_eq!(result, 89);
    }

    #[test]
    fn jit_fibonacci_matches_interp() {
        let prog_str = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1";
        let prog: Vec<&str> = prog_str.split_whitespace().collect();

        for n in [5, 10, 11, 15, 20] {
            let mut interp_args = vec![
                interp::Box::Int(1),
                interp::Box::Int(1),
                interp::Box::Int(n),
            ];
            let interp_result = interp::interpret(&prog, &mut interp_args);
            let expected = interp::repr_stack(&interp_result);

            let mut jit = JitTiny3Interp::new();
            let mut jit_args = vec![1i64, 1, n];
            let jit_result = jit.run(&prog, &mut jit_args);

            assert_eq!(jit_result.to_string(), expected, "fib({n}) mismatch");
        }
    }

    #[test]
    fn jit_countdown() {
        let prog: Vec<&str> = "{ #1 #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![5i64];
        jit.run(&prog, &mut args);
        assert_eq!(args[0], 0);
    }

    #[test]
    fn jit_float_arithmetic() {
        // Float countdown: multiply by 0.5 each iteration
        // Program: { #1 0.5 MUL ->#1 #2 1 SUB ->#2 #2 }
        // After 10 iterations with args=[8.0, 10]: 8.0 * 0.5^10
        let prog: Vec<&str> = "{ #1 0.5 MUL ->#1 #2 1 SUB ->#2 #2 }"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![interp::Box::Float(8.0), interp::Box::Int(10)];
        jit.run_typed(&prog, &mut args);
        let result = args[0].as_float();
        let expected = 8.0 * 0.5f64.powi(10);
        assert!(
            (result - expected).abs() < 1e-10,
            "expected ~{expected}, got {result}"
        );
        assert_eq!(args[1].as_int(), 0);
    }

    #[test]
    fn jit_mixed_int_float() {
        // Mixed arithmetic: #1(float) + #2(int) should produce float
        // Program: { #1 #2 ADD ->#1 #3 1 SUB ->#3 #3 }
        // args = [1.5, 2, 5]  → after each iter: 3.5, 5.5, 7.5, 9.5, 11.5
        let prog: Vec<&str> = "{ #1 #2 ADD ->#1 #3 1 SUB ->#3 #3 }"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![
            interp::Box::Float(1.5),
            interp::Box::Int(2),
            interp::Box::Int(5),
        ];
        jit.run_typed(&prog, &mut args);
        let result = args[0].as_float();
        assert!(
            (result - 11.5).abs() < 1e-10,
            "expected 11.5, got {result}"
        );
        assert_eq!(args[2].as_int(), 0);
    }
}
