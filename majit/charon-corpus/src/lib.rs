//! Charon fixture corpus: representative shapes from issue #97.
//!
//! 1. `straight_line_add` — straight-line interpreter-shaped function.
//! 2. `branch_loop_sum`   — branch + loop, like opcode dispatch fragments.
//! 3. `strategy_dispatch` — enum-as-strategy (dict-strategy stand-in).
//! 4. `desugar_mix`       — `?`, `match`, and iterator desugaring together.

#![allow(dead_code)]

pub type PyResult<T> = Result<T, &'static str>;

// --- 1. Straight-line ---------------------------------------------------

#[inline(never)]
pub fn straight_line_add(a: i64, b: i64, c: i64) -> i64 {
    let s = a + b;
    let t = s * 2;
    t + c
}

// --- 2. Branch + loop ---------------------------------------------------

#[inline(never)]
pub fn branch_loop_sum(slice: &[i64], threshold: i64) -> i64 {
    let mut acc: i64 = 0;
    for &v in slice {
        if v > threshold {
            acc += v;
        } else {
            acc -= v;
        }
    }
    acc
}

// --- 3. Strategy dispatch (dict-strategy stand-in) ----------------------

pub enum Strategy {
    Empty,
    IntKeyed { len: usize },
    StrKeyed { len: usize, capacity: usize },
}

#[inline(never)]
pub fn strategy_len(s: &Strategy) -> usize {
    match s {
        Strategy::Empty => 0,
        Strategy::IntKeyed { len } => *len,
        Strategy::StrKeyed { len, capacity: _ } => *len,
    }
}

// --- 4. Desugar mix: ?, match, iterator --------------------------------

pub enum Token {
    Add(i64),
    Sub(i64),
    Halt,
}

fn parse_one(raw: i64) -> PyResult<Token> {
    match raw {
        i64::MIN => Ok(Token::Halt),
        0 => Err("halt-zero forbidden"),
        v if v > 0 => Ok(Token::Add(v)),
        v => Ok(Token::Sub(-v)),
    }
}

#[inline(never)]
pub fn desugar_mix(input: &[i64]) -> PyResult<i64> {
    let mut acc: i64 = 0;
    for &raw in input.iter() {
        let tok = parse_one(raw)?;
        match tok {
            Token::Add(v) => acc += v,
            Token::Sub(v) => acc -= v,
            Token::Halt => break,
        }
    }
    Ok(acc)
}

// --- 5. Tuple round-trip: construct a tuple, read .0/.1 in same fn ------
//
// Exercises `Rvalue::Aggregate` for a *non-Adt* (tuple) value paired
// with `Field` projection reads of that same local. The lowering must
// emit a `__pos_<idx>` `FieldRead` symmetric to the construction-side
// `FieldWrite` chain rather than collapsing every `.N` to the base.

#[inline(never)]
pub fn tuple_roundtrip(a: i64, b: i64) -> i64 {
    let pair = (a + b, a - b);
    pair.0 * pair.1
}
