//! The regex program: an authoring tree, and the `NodeRec` graph it lowers to.
//!
//! Ported from "An Efficient and Elegant Regular Expression Matcher in Python".
//! A regex is a tree, and matching carries *marks* through it — so a mark is
//! state that lives in the node, not in the matcher.  `empty` (does this
//! subexpression accept the empty string?) is a pure function of the shape, so
//! `lower` computes it once, bottom up, and the matcher only ever reads it.

/// Node kinds.  Stored as a `u8` tag so a `NodeRec` field read is what the
/// matcher dispatches on.
pub const KIND_CHAR: u8 = 0;
pub const KIND_EPSILON: u8 = 1;
pub const KIND_ALTERNATIVE: u8 = 2;
pub const KIND_SEQUENCE: u8 = 3;
pub const KIND_REPETITION: u8 = 4;

/// The lowered representation the matcher walks.  `kind`, `ch`, `empty`,
/// `left` and `right` are all fixed once lowered; `marked` is the single
/// mutable bit the algorithm shifts around.
///
/// The declaration below names four of the five fixed fields.  `empty` is
/// immutable as well and is left undeclared for now.
#[repr(C)]
#[majit_macros::jit_immutable_fields("left", "right", "kind", "ch")]
pub struct NodeRec {
    pub kind: u8,
    pub ch: u8,
    pub empty: u8,
    pub marked: u8,
    pub left: *mut NodeRec,
    pub right: *mut NodeRec,
}

/// The authoring surface.  Trees are built with this and immediately lowered;
/// nothing matches against it.
pub enum Node {
    Char(u8),
    Epsilon,
    Alternative(Box<Node>, Box<Node>),
    Sequence(Box<Node>, Box<Node>),
    Repetition(Box<Node>),
}

pub fn ch(c: u8) -> Box<Node> {
    Box::new(Node::Char(c))
}
pub fn epsilon() -> Box<Node> {
    Box::new(Node::Epsilon)
}
pub fn alt(l: Box<Node>, r: Box<Node>) -> Box<Node> {
    Box::new(Node::Alternative(l, r))
}
pub fn seq(l: Box<Node>, r: Box<Node>) -> Box<Node> {
    Box::new(Node::Sequence(l, r))
}
pub fn rep(n: Box<Node>) -> Box<Node> {
    Box::new(Node::Repetition(n))
}

/// Lower once, before the matcher runs.  Every `Node` becomes its own
/// `NodeRec`: nodes carry marks, so an instance may never be shared — the
/// twenty `(a|b)` groups of `(a|b){20}` are twenty distinct records.
///
/// The result is leaked.  The graph is the program, and a program the JIT has
/// specialized on must stay at the same address for the life of the process.
pub fn lower(n: &Node) -> *mut NodeRec {
    let null = core::ptr::null_mut();
    let rec = match n {
        Node::Char(c) => NodeRec {
            kind: KIND_CHAR,
            ch: *c,
            empty: 0,
            marked: 0,
            left: null,
            right: null,
        },
        Node::Epsilon => NodeRec {
            kind: KIND_EPSILON,
            ch: 0,
            empty: 1,
            marked: 0,
            left: null,
            right: null,
        },
        Node::Alternative(l, r) => {
            let (lp, rp) = (lower(l), lower(r));
            let empty = unsafe { (*lp).empty | (*rp).empty };
            NodeRec {
                kind: KIND_ALTERNATIVE,
                ch: 0,
                empty,
                marked: 0,
                left: lp,
                right: rp,
            }
        }
        Node::Sequence(l, r) => {
            let (lp, rp) = (lower(l), lower(r));
            let empty = unsafe { (*lp).empty & (*rp).empty };
            NodeRec {
                kind: KIND_SEQUENCE,
                ch: 0,
                empty,
                marked: 0,
                left: lp,
                right: rp,
            }
        }
        Node::Repetition(inner) => {
            let ip = lower(inner);
            NodeRec {
                kind: KIND_REPETITION,
                ch: 0,
                empty: 1,
                marked: 0,
                left: ip,
                right: null,
            }
        }
    };
    Box::leak(Box::new(rec)) as *mut NodeRec
}

// ── fixtures ───────────────────────────────────────────────────────────────

/// `abc`
pub fn abc() -> Box<Node> {
    seq(seq(ch(b'a'), ch(b'b')), ch(b'c'))
}
/// `a|b`
pub fn ab() -> Box<Node> {
    alt(ch(b'a'), ch(b'b'))
}

/// `(a|b)*a(a|b){n}a(a|b)*` — the benchmark regex of "A JIT for Regular
/// Expression Matching" — built BALANCED.  Association is free: it does not
/// change the language, only the tree depth (at `n = 20`, 8 balanced against
/// 26 left-associated).
pub fn bench_regex(n: usize) -> Box<Node> {
    let mut parts: Vec<Box<Node>> = Vec::new();
    parts.push(rep(ab()));
    parts.push(ch(b'a'));
    for _ in 0..n {
        parts.push(ab());
    }
    parts.push(ch(b'a'));
    parts.push(rep(ab()));
    build_balanced(parts)
}

#[expect(
    clippy::vec_box,
    reason = "the builders hand out Box<Node> because Node is recursive; unboxing here would only move the boxing to the call sites"
)]
fn build_balanced(mut xs: Vec<Box<Node>>) -> Box<Node> {
    if xs.len() == 1 {
        return xs.pop().unwrap();
    }
    let rhs = xs.split_off(xs.len() / 2);
    seq(build_balanced(xs), build_balanced(rhs))
}

/// The same regex, left-associated, for the depth comparison.
pub fn bench_regex_left(n: usize) -> Box<Node> {
    let mut node = rep(ab());
    node = seq(node, ch(b'a'));
    for _ in 0..n {
        node = seq(node, ab());
    }
    node = seq(node, ch(b'a'));
    seq(node, rep(ab()))
}

/// Deepest path through the graph.  Nodes are never shared, so this is a plain
/// tree walk.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "every NodeRec is leaked by `lower` and lives for the process"
)]
pub fn depth(n: *mut NodeRec) -> usize {
    let mut d = 0;
    let (l, r) = unsafe { ((*n).left, (*n).right) };
    if !l.is_null() {
        d = d.max(depth(l));
    }
    if !r.is_null() {
        d = d.max(depth(r));
    }
    d + 1
}

/// Records reachable from `n`.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "every NodeRec is leaked by `lower` and lives for the process"
)]
pub fn count(n: *mut NodeRec) -> usize {
    let mut c = 1;
    let (l, r) = unsafe { ((*n).left, (*n).right) };
    if !l.is_null() {
        c += count(l);
    }
    if !r.is_null() {
        c += count(r);
    }
    c
}

/// A random `a`/`b` string forced NOT to match `(a|b)*a(a|b){n}a(a|b)*`.
///
/// That regex matches iff some pair of `a`s sits exactly `n` apart, i.e.
/// `s[i] == s[i + n + 1] == 'a'`.  A random string of any length almost surely
/// contains such a pair, so a plain random string is not a non-matching input;
/// clearing the pairs left to right is what makes one.
///
/// A non-matching input is what the benchmark wants: the matcher then has to
/// look at every character, and no early exit hides the per-character cost.
pub fn nonmatching(len: usize, n: usize, mut seed: u64) -> Vec<u8> {
    let mut s = Vec::with_capacity(len);
    for _ in 0..len {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push(if (seed >> 33) & 1 == 0 { b'a' } else { b'b' });
    }
    let d = n + 1;
    for i in 0..len.saturating_sub(d) {
        if s[i] == b'a' && s[i + d] == b'a' {
            s[i + d] = b'b';
        }
    }
    s
}

/// The correctness vectors: `(regex, input, should match)`.  Shared by `main`
/// and the tests so the two can never drift.
pub fn vectors() -> Vec<(Box<Node>, &'static str, bool)> {
    // `a|b|c`
    let abc_alt = || alt(alt(ch(b'a'), ch(b'b')), ch(b'c'));
    // `((abc)*|(abcd))(d|e)`
    let tricky = || {
        seq(
            alt(rep(abc()), seq(abc(), ch(b'd'))),
            alt(ch(b'd'), ch(b'e')),
        )
    };
    vec![
        (abc_alt(), "a", true),
        (abc_alt(), "b", true),
        (abc_alt(), "c", true),
        (abc_alt(), "d", false),
        (abc_alt(), "", false),
        (abc_alt(), "ab", false),
        // `(a|b|c)*`
        (rep(abc_alt()), "abcbac", true),
        (rep(abc_alt()), "", true),
        (rep(abc_alt()), "abd", false),
        (rep(abc_alt()), "a", true),
        // `abc`
        (abc(), "abc", true),
        (abc(), "abcd", false),
        (abc(), "ab", false),
        (abc(), "", false),
        // `((abc)*|(abcd))(d|e)`
        (tricky(), "abcabcabcd", true),
        (tricky(), "abcd", true),
        // `abcd` from the right alternative, then `e`
        (tricky(), "abcde", true),
        (tricky(), "abcdf", false),
        (tricky(), "abcabcd", true),
        // `(abc)*` accepts the empty string
        (tricky(), "d", true),
        (tricky(), "e", true),
        // epsilon
        (epsilon(), "", true),
        (epsilon(), "a", false),
        // `(a|b)*a(a|b){2}a(a|b)*`
        (bench_regex(2), "aaaa", true),
        (bench_regex(2), "abba", true),
        (bench_regex(2), "aba", false),
        (bench_regex(2), "babbab", true),
        (bench_regex(2), "bbbb", false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_table_is_complete() {
        assert_eq!(vectors().len(), 28);
    }

    #[test]
    fn test_bench_regex_node_count() {
        assert_eq!(count(lower(&bench_regex(20))), 93);
        assert_eq!(count(lower(&bench_regex_left(20))), 93);
    }

    #[test]
    fn test_association_changes_depth_only() {
        assert_eq!(depth(lower(&bench_regex(20))), 8);
        assert_eq!(depth(lower(&bench_regex_left(20))), 26);
    }

    #[test]
    fn test_lower_computes_empty_bottom_up() {
        let root = lower(&seq(rep(ch(b'a')), epsilon()));
        unsafe {
            assert_eq!((*root).empty, 1);
            assert_eq!((*(*root).left).empty, 1);
            assert_eq!((*(*(*root).left).left).empty, 0);
        }
    }

    /// Marks live in the nodes, so a shared instance would be a shared mark.
    #[test]
    fn test_lower_never_shares_a_node() {
        let root = lower(&bench_regex(20));
        let mut seen = Vec::new();
        collect(root, &mut seen);
        let n = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n);
    }

    fn collect(n: *mut NodeRec, out: &mut Vec<usize>) {
        out.push(n as usize);
        let (l, r) = unsafe { ((*n).left, (*n).right) };
        if !l.is_null() {
            collect(l, out);
        }
        if !r.is_null() {
            collect(r, out);
        }
    }
}
