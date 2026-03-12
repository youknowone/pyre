/// Interpreter for tiny3_hotpath — direct translation of rpython/jit/tl/tiny3_hotpath.py.
///
/// A word-based language identical to tiny2 but with IntBox/FloatBox instead of
/// IntBox/StrBox. Arithmetic on mixed int/float types automatically casts to float.
///
///    6 7 ADD              => 13
///    3.8 1 ADD            => 4.8
///    3.8                  => 3.8

/// A boxed value — either a known integer or a float.
#[derive(Clone, Debug)]
pub enum Box {
    Int(i64),
    Float(f64),
}

impl Box {
    pub fn as_int(&self) -> i64 {
        match self {
            Box::Int(v) => *v,
            Box::Float(v) => *v as i64,
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Box::Int(v) => *v as f64,
            Box::Float(v) => *v,
        }
    }

    pub fn as_str(&self) -> String {
        match self {
            Box::Int(v) => v.to_string(),
            Box::Float(v) => format!("{v}"),
        }
    }
}

fn parse_int(s: &str, start: usize) -> Option<i64> {
    let s = &s[start..];
    if s.is_empty() {
        return None;
    }
    let mut res: i64 = 0;
    for c in s.chars() {
        let d = c as i64 - '0' as i64;
        if !(0..=9).contains(&d) {
            return None;
        }
        res = res * 10 + d;
    }
    Some(res)
}

fn op2(stack: &mut Vec<Box>, op: &str) {
    let y = stack.pop().unwrap();
    let x = stack.pop().unwrap();
    let result = match (&x, &y) {
        (Box::Int(ix), Box::Int(iy)) => {
            let r = match op {
                "ADD" => ix + iy,
                "SUB" => ix - iy,
                "MUL" => ix * iy,
                _ => unreachable!(),
            };
            Box::Int(r)
        }
        _ => {
            let fx = x.as_float();
            let fy = y.as_float();
            let r = match op {
                "ADD" => fx + fy,
                "SUB" => fx - fy,
                "MUL" => fx * fy,
                _ => unreachable!(),
            };
            Box::Float(r)
        }
    };
    stack.push(result);
}

/// Interpret a word-based program with the given arguments.
pub fn interpret(bytecode: &[&str], args: &mut [Box]) -> Vec<Box> {
    let mut stack: Vec<Box> = Vec::new();
    let mut loops: Vec<usize> = Vec::new();
    let mut pos: usize = 0;

    while pos < bytecode.len() {
        let opcode = bytecode[pos];
        pos += 1;

        if opcode == "ADD" || opcode == "SUB" || opcode == "MUL" {
            op2(&mut stack, opcode);
        } else if opcode.starts_with('#') {
            let n = parse_int(opcode, 1).unwrap() as usize;
            stack.push(args[n - 1].clone());
        } else if opcode.starts_with("->#") {
            let n = parse_int(opcode, 3).unwrap() as usize;
            let val = stack.pop().unwrap();
            args[n - 1] = val;
        } else if opcode == "{" {
            loops.push(pos);
        } else if opcode == "}" {
            let flag = stack.pop().unwrap();
            if flag.as_int() == 0 {
                loops.pop();
            } else {
                pos = *loops.last().unwrap();
            }
        } else {
            // Try to parse as integer, then float
            if let Some(v) = parse_int(opcode, 0) {
                stack.push(Box::Int(v));
            } else if let Ok(v) = opcode.parse::<f64>() {
                stack.push(Box::Float(v));
            }
            // Ignore unparseable words (they'd be string words, but tiny3 only
            // supports int/float)
        }
    }

    stack
}

/// Format stack contents as space-separated string.
pub fn repr_stack(stack: &[Box]) -> String {
    stack
        .iter()
        .map(|b| b.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_int() {
        let prog: Vec<&str> = "6 7 ADD".split_whitespace().collect();
        let result = interpret(&prog, &mut []);
        assert_eq!(repr_stack(&result), "13");
    }

    #[test]
    fn test_mul() {
        let prog: Vec<&str> = "7 5 MUL".split_whitespace().collect();
        let result = interpret(&prog, &mut []);
        assert_eq!(repr_stack(&result), "35");
    }

    #[test]
    fn test_add_with_int_arg() {
        let prog: Vec<&str> = "#1 5 ADD".split_whitespace().collect();
        let result = interpret(&prog, &mut [Box::Int(20)]);
        assert_eq!(repr_stack(&result), "25");
    }

    #[test]
    fn test_add_with_float_arg() {
        let prog: Vec<&str> = "#1 5 ADD".split_whitespace().collect();
        let result = interpret(&prog, &mut [Box::Float(3.2)]);
        assert_eq!(repr_stack(&result), "8.2");
    }

    #[test]
    fn test_float_literal() {
        let prog: Vec<&str> = "3.8 1 ADD".split_whitespace().collect();
        let result = interpret(&prog, &mut []);
        assert_eq!(repr_stack(&result), "4.8");
    }

    #[test]
    fn test_factorial() {
        // "The factorial of #1 is 1 { #1 MUL #1 1 SUB ->#1 #1 }"
        // tiny3 ignores non-numeric words, so "The", "factorial", "of", "is" are dropped.
        // Use the numeric-only version:
        let prog: Vec<&str> = "1 { #1 MUL #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let result = interpret(&prog, &mut [Box::Int(5)]);
        assert_eq!(repr_stack(&result), "120");
    }

    #[test]
    fn test_fibonacci() {
        let prog: Vec<&str> = "{ #1 #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 }"
            .split_whitespace()
            .collect();
        let result = interpret(&prog, &mut [Box::Int(1), Box::Int(1), Box::Int(10)]);
        assert_eq!(repr_stack(&result), "1 1 2 3 5 8 13 21 34 55");
    }

    #[test]
    fn test_fibonacci_single() {
        let prog: Vec<&str> = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1"
            .split_whitespace()
            .collect();
        let result = interpret(&prog, &mut [Box::Int(1), Box::Int(1), Box::Int(11)]);
        assert_eq!(repr_stack(&result), "89");
    }

    #[test]
    fn test_loop_countdown() {
        let prog: Vec<&str> = "{ #1 #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let result = interpret(&prog, &mut [Box::Int(5)]);
        assert_eq!(repr_stack(&result), "5 4 3 2 1");
    }
}
