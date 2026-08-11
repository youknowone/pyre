/// Reference interpreter for a two-tape Brainfuck-like language.
///
/// Two independent tapes `a` and `b`, each with its own pointer:
///   `>` `<` `+` `-`   move/mutate tape `a`
///   `}` `{` `*` `/`   move/mutate tape `b`
///   `[` `]`           loop on tape `a`'s current cell
///
/// `interpret` returns the sum of every cell across both tapes so the JIT
/// result can be compared against this oracle with a single integer.

pub const TAPE_SIZE: usize = 8;

pub fn interpret(code: &[u8]) -> i64 {
    let mut a = vec![0i64; TAPE_SIZE];
    let mut b = vec![0i64; TAPE_SIZE];
    let mut pa: usize = 0;
    let mut pb: usize = 0;
    let mut pc: usize = 0;

    while pc < code.len() {
        pc = step(code, pc, &mut a, &mut b, &mut pa, &mut pb);
    }

    a.iter().sum::<i64>() + b.iter().sum::<i64>()
}

fn step(
    code: &[u8],
    pc: usize,
    a: &mut [i64],
    b: &mut [i64],
    pa: &mut usize,
    pb: &mut usize,
) -> usize {
    match code[pc] {
        b'>' => *pa += 1,
        b'<' => *pa -= 1,
        b'+' => a[*pa] += 1,
        b'-' => a[*pa] -= 1,
        b'}' => *pb += 1,
        b'{' => *pb -= 1,
        b'*' => b[*pb] += 1,
        b'/' => b[*pb] -= 1,
        b'[' => {
            if a[*pa] == 0 {
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
        }
        b']' if a[*pa] != 0 => {
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
        _ => {}
    }
    pc + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_tape_loop_sum() {
        // a[0]=10; loop 10x: a[0]-=1, a[1]+=1, b[0]+=1, b[1]+=1.
        // final: a[1]=10, b[0]=10, b[1]=10 → sum 30.
        assert_eq!(interpret(b"++++++++++[->+<*}*{]"), 30);
    }
}
