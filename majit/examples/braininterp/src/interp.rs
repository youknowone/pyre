/// Interpreter for Brainfuck — direct translation of rpython/jit/tl/braininterp.py.
///
/// Tape-based interpreter with 30000 cells, byte-sized values.
/// Operations: > < + - . , [ ]

const TAPE_SIZE: usize = 30000;

pub fn interpret(code: &[u8]) -> String {
    let mut tape = vec![0u8; TAPE_SIZE];
    let mut pointer: usize = 0;
    let mut output = String::new();
    let mut pc: usize = 0;

    while pc < code.len() {
        pc = interp_char(code, pc, &mut tape, &mut pointer, &mut output);
    }

    output
}

fn interp_char(
    code: &[u8],
    pc: usize,
    tape: &mut [u8],
    pointer: &mut usize,
    output: &mut String,
) -> usize {
    let ch = code[pc];
    if ch == b'>' {
        *pointer += 1;
    } else if ch == b'<' {
        *pointer -= 1;
    } else if ch == b'+' {
        tape[*pointer] = tape[*pointer].wrapping_add(1);
    } else if ch == b'-' {
        tape[*pointer] = tape[*pointer].wrapping_sub(1);
    } else if ch == b'.' {
        output.push(tape[*pointer] as char);
    } else if ch == b',' {
        // Input not supported in this port; set to 0.
        tape[*pointer] = 0;
    } else if ch == b'[' {
        if tape[*pointer] == 0 {
            let mut need: i32 = 1;
            let mut p = pc + 1;
            while need > 0 {
                if code[p] == b']' {
                    need -= 1;
                } else if code[p] == b'[' {
                    need += 1;
                }
                p += 1;
            }
            return p;
        }
    } else if ch == b']' {
        if tape[*pointer] != 0 {
            let mut need: i32 = 1;
            let mut p = pc - 1;
            while need > 0 {
                if code[p] == b']' {
                    need += 1;
                } else if code[p] == b'[' {
                    need -= 1;
                }
                if need > 0 {
                    p -= 1;
                }
            }
            return p + 1;
        }
    }
    pc + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increment() {
        let output = interpret(b"+++");
        // After +++, cell 0 should be 3 (but no output)
        assert_eq!(output.len(), 0); // no '.' so no output
    }

    #[test]
    fn test_simple_output() {
        // Set cell 0 to 65 ('A') and output it
        // 65 = 8*8 + 1: ++++++++[>++++++++<-]>+.
        let output = interpret(b"++++++++[>++++++++<-]>+.");
        assert_eq!(output, "A");
    }

    #[test]
    fn test_hello_world() {
        let code = b"++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
        let output = interpret(code);
        assert_eq!(output, "Hello World!\n");
    }

    #[test]
    fn test_move_value() {
        // Move cell 0 to cell 1: [->+<]
        // Start with cell0=5
        let output = interpret(b"+++++[->+<]");
        // No output, but the loop should complete
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn test_nested_loop() {
        // Multiply: cell0 * cell1 -> cell2
        // Set cell0=3, cell1=4: +++>++++<[->[->>+<<]>>[-<+<+>>]<<<]
        // Result: cell2 = 12, output nothing
        let output = interpret(b"+++>++++<[->[->>+<<]>>[-<+<+>>]<<<]");
        assert_eq!(output.len(), 0);
    }
}
