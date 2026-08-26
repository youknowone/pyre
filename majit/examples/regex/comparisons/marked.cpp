// The marked-regex matcher in C++ — the "C++ (Sebastian Fischer)" row of
// "A JIT for Regular Expression Matching" (2010), which reported 750,000
// chars/s on a 2.26 GHz Core 2 Duo.
//
// This is a transcription of the crate's `src/interp.rs`, not a rewrite: the
// same five node kinds, the same `NodeRec` field set, the same `shift` arms in
// the same order, `empty` computed once while the tree is built, and `&` / `|`
// where part 1 of the blog series has `and` / `or` — the substitution part 2
// makes so the body carries no short-circuit branch.  The tree is built
// BALANCED, the same way `regex.rs::build_balanced` builds it, and the input
// comes from the same LCG and the same non-matching fixup, so the row is
// comparable with the Rust rows character for character.
//
// Two ways this row could lie, and what is done about each:
//
//   * The matcher exits early.  It cannot: `matches` is a `for` over the whole
//     input with no `break`, and the answer is checked to be "no match" — a
//     non-matching input is the point, because then no early exit hides the
//     per-character cost.  If the answer ever comes back "match" the program
//     exits non-zero rather than posting the number.
//   * The optimizer deletes the work.  `sink` accumulates every round's answer
//     and is printed, and every mark store lands in a heap node the tree still
//     points at, so the stores are not dead.  The stronger check is external:
//     the reported node-visit rate must scale as 1 / (node count), so running
//     this at `n = 2` (21 nodes) must come out roughly 4.4x faster per
//     character than at `n = 20` (93 nodes).  A rate that does not move with
//     the node count means the walk was folded away, which at -O2 with a
//     runtime-built tree would be a surprise worth chasing.  `run.sh` runs that
//     control.
//
// `marked --verify <length> [n]` is the correctness gate: it prints a line that
// every port must print identically — the input digest, the answers to a fixed
// battery, and a digest of all 93 marks left after scanning the benchmark
// input.  See the `verify` section below for why each part is there.  It
// catches the third way this row could lie: a matcher that lost an arm and
// answers "no match" to everything would pass the check above and post a
// number.
//
// Usage: marked <length> <repeats> [n]      (n defaults to 20)
// Prints one line on stdout: `cpp <chars_per_second>` (median of the rounds).
// Per-round detail goes to stderr as `round <i> <chars_per_second>`.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum Kind : uint8_t {
    KIND_CHAR = 0,
    KIND_EPSILON = 1,
    KIND_ALTERNATIVE = 2,
    KIND_SEQUENCE = 3,
    KIND_REPETITION = 4,
};

// `kind`, `ch`, `empty`, `left` and `right` are fixed once the tree is built;
// `marked` is the single mutable bit the algorithm shifts around.  Same field
// set and same order as the Rust `NodeRec`.
struct Node {
    uint8_t kind;
    uint8_t ch;
    uint8_t empty;
    uint8_t marked;
    Node *left;
    Node *right;
};

// Nodes are leaked, exactly as `regex.rs::lower` leaks them: the graph is the
// program and lives for the process.  Children are allocated before their
// parent so the address order matches the Rust side's post-order `lower`.
static Node *mk(uint8_t kind, uint8_t ch, uint8_t empty, Node *l, Node *r) {
    Node *n = new Node;
    n->kind = kind;
    n->ch = ch;
    n->empty = empty;
    n->marked = 0;
    n->left = l;
    n->right = r;
    return n;
}

static Node *node_char(uint8_t c) { return mk(KIND_CHAR, c, 0, nullptr, nullptr); }
static Node *node_epsilon() { return mk(KIND_EPSILON, 0, 1, nullptr, nullptr); }
static Node *node_alt(Node *l, Node *r) {
    return mk(KIND_ALTERNATIVE, 0, (uint8_t)(l->empty | r->empty), l, r);
}
static Node *node_seq(Node *l, Node *r) {
    return mk(KIND_SEQUENCE, 0, (uint8_t)(l->empty & r->empty), l, r);
}
static Node *node_rep(Node *inner) { return mk(KIND_REPETITION, 0, 1, inner, nullptr); }

// `a|b`.  A fresh pair every call: marks live in the nodes, so an instance may
// never be shared — the twenty `(a|b)` groups of `(a|b){20}` are twenty
// distinct records.
static Node *ab() { return node_alt(node_char('a'), node_char('b')); }

// Balanced association, matching `regex.rs::build_balanced`: the left half is
// `xs[0 .. len/2]` and the right half is `xs[len/2 ..]`.  C++ leaves the
// evaluation order of function arguments unspecified, so the two halves are
// built into named locals first — the association is what fixes the node
// addresses, and a differently ordered allocation would be a different memory
// layout for the same language.
static Node *build_balanced(std::vector<Node *> &xs, size_t lo, size_t hi) {
    if (hi - lo == 1) {
        return xs[lo];
    }
    size_t mid = lo + (hi - lo) / 2;
    Node *l = build_balanced(xs, lo, mid);
    Node *r = build_balanced(xs, mid, hi);
    return node_seq(l, r);
}

// `(a|b)*a(a|b){n}a(a|b)*`, the benchmark regex of the post.
static Node *bench_regex(size_t n) {
    std::vector<Node *> parts;
    parts.push_back(node_rep(ab()));
    parts.push_back(node_char('a'));
    for (size_t i = 0; i < n; i++) {
        parts.push_back(ab());
    }
    parts.push_back(node_char('a'));
    parts.push_back(node_rep(ab()));
    return build_balanced(parts, 0, parts.size());
}

static size_t count_nodes(Node *n) {
    size_t c = 1;
    if (n->left) c += count_nodes(n->left);
    if (n->right) c += count_nodes(n->right);
    return c;
}

// Push `mark` into `n` for input character `c`, and answer the mark that comes
// out on the right.  The mark left inside `n` is the one this call computed.
//
// `&` and `|`, never `&&` and `||`: both sides must run, and the point of the
// substitution is that the body has no conditional branch in it at all.
static int64_t shift(Node *n, int64_t c, int64_t mark) {
    int64_t m;
    switch (n->kind) {
    case KIND_CHAR:
        m = mark & (int64_t)((int64_t)n->ch == c);
        break;
    case KIND_EPSILON:
        m = 0;
        break;
    case KIND_ALTERNATIVE: {
        // Sequenced by hand: `a | b` does not order its operands in C++.  The
        // two subtrees are disjoint so it could not change the answer, but the
        // Rust reads left then right and this should diff against it cleanly.
        int64_t ml = shift(n->left, c, mark);
        int64_t mr = shift(n->right, c, mark);
        m = ml | mr;
        break;
    }
    case KIND_REPETITION:
        m = shift(n->left, c, mark | (int64_t)n->marked);
        break;
    case KIND_SEQUENCE: {
        // The left mark from the PREVIOUS character is what enters the right
        // side, so read it before `shift` overwrites it.
        int64_t old_marked_left = (int64_t)n->left->marked;
        int64_t marked_left = shift(n->left, c, mark);
        int64_t marked_right =
            shift(n->right, c, old_marked_left | (mark & (int64_t)n->left->empty));
        m = (marked_left & (int64_t)n->right->empty) | marked_right;
        break;
    }
    default:
        m = 0;
        break;
    }
    n->marked = (uint8_t)m;
    return m;
}

// Clear every mark, so the graph can match another string.
static void reset(Node *n) {
    n->marked = 0;
    if (n->left) reset(n->left);
    if (n->right) reset(n->right);
}

// Shift one mark in from the left for `s[0]`, then shift the marks already
// inside the graph along for every remaining character.  No `break`: every
// character is looked at, whatever the answer turns out to be.
static bool matches(Node *root, const uint8_t *s, size_t len) {
    if (len == 0) {
        return root->empty != 0;
    }
    int64_t result = shift(root, (int64_t)s[0], 1);
    for (size_t i = 1; i < len; i++) {
        result = shift(root, (int64_t)s[i], 0);
    }
    reset(root);
    return result != 0;
}

// A random `a`/`b` string forced NOT to match `(a|b)*a(a|b){n}a(a|b)*`.
//
// Byte for byte the generator in `regex.rs::nonmatching`: the same LCG
// constants, the same bit picked out of the state, and the same left-to-right
// fixup.  That regex matches iff some pair of `a`s sits exactly `n + 1` apart,
// and a random string almost surely has one, so clearing those pairs is what
// makes a non-matching input.
static std::vector<uint8_t> nonmatching(size_t len, size_t n, uint64_t seed) {
    std::vector<uint8_t> s;
    s.reserve(len);
    for (size_t i = 0; i < len; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        s.push_back(((seed >> 33) & 1) == 0 ? (uint8_t)'a' : (uint8_t)'b');
    }
    size_t d = n + 1;
    if (len > d) {
        for (size_t i = 0; i + d < len; i++) {
            if (s[i] == 'a' && s[i + d] == 'a') {
                s[i + d] = 'b';
            }
        }
    }
    return s;
}

// A digest of the input, so `run.sh` can prove every port scanned the same
// bytes.  Any hash would do; FNV-1a is four lines in all four languages.
static uint64_t fnv1a64(const std::vector<uint8_t> &data) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint8_t b : data) {
        h = (h ^ (uint64_t)b) * 0x100000001b3ULL;
    }
    return h;
}

// ── verify ─────────────────────────────────────────────────────────────────
//
// `--verify` is the correctness gate, and it is built so that the four ports
// can be checked against EACH OTHER and against the crate's `interp::matches`,
// not merely each against its own copy of an expectation.  It prints one line,
// and every port must print the same one:
//
//   verify nodes=.. input_fnv1a=.. head=.. tail=.. answers=.. marks=..
//
//   * `input_fnv1a`, `head`, `tail` — the bytes this port generated.  A port
//     whose LCG is off by one wrapping multiply still produces a plausible
//     `a`/`b` string, and a chars/s number taken over different bytes is not a
//     row in the same table.
//   * `answers` — one bit per case of a fixed battery: the 28 vectors of
//     `regex.rs::vectors()`, then four cases at the benchmark's own scale (the
//     non-matching input, and the same input with a matching pair of `a`s
//     planted at the front, in the middle, and hard against the end).  The
//     planted cases catch a matcher that lost an arm and answers "no" to
//     everything — which a non-matching benchmark alone cannot catch — and the
//     last one only matches if the final byte was read.
//   * `marks` — a digest of all 93 `marked` bits left in the tree after
//     scanning the benchmark input, taken BEFORE `reset` clears them.  This is
//     the strong one: it compares the whole state of the computation after a
//     million characters, so two ports agreeing on it are doing the same work
//     and not merely arriving at the same boolean.
//
// The 28 vectors are also checked against their expected answers here, so a
// port that agrees with the others but disagrees with `regex.rs` still fails.

static Node *re_abc() {  // `abc`
    return node_seq(node_seq(node_char('a'), node_char('b')), node_char('c'));
}
static Node *re_abc_alt() {  // `a|b|c`
    return node_alt(node_alt(node_char('a'), node_char('b')), node_char('c'));
}
static Node *re_tricky() {  // `((abc)*|(abcd))(d|e)`
    Node *l = node_alt(node_rep(re_abc()), node_seq(re_abc(), node_char('d')));
    Node *r = node_alt(node_char('d'), node_char('e'));
    return node_seq(l, r);
}

enum Which { W_ABC_ALT, W_REP_ABC_ALT, W_ABC, W_TRICKY, W_EPSILON, W_BENCH2 };

static Node *build(Which w) {
    switch (w) {
    case W_ABC_ALT: return re_abc_alt();
    case W_REP_ABC_ALT: return node_rep(re_abc_alt());
    case W_ABC: return re_abc();
    case W_TRICKY: return re_tricky();
    case W_EPSILON: return node_epsilon();
    default: return bench_regex(2);
    }
}

// `regex.rs::vectors()`, in its order.
struct VecCase {
    Which w;
    const char *s;
    bool want;
};
static const VecCase VECTORS[] = {
    {W_ABC_ALT, "a", true},          {W_ABC_ALT, "b", true},
    {W_ABC_ALT, "c", true},          {W_ABC_ALT, "d", false},
    {W_ABC_ALT, "", false},          {W_ABC_ALT, "ab", false},
    {W_REP_ABC_ALT, "abcbac", true}, {W_REP_ABC_ALT, "", true},
    {W_REP_ABC_ALT, "abd", false},   {W_REP_ABC_ALT, "a", true},
    {W_ABC, "abc", true},            {W_ABC, "abcd", false},
    {W_ABC, "ab", false},            {W_ABC, "", false},
    {W_TRICKY, "abcabcabcd", true},  {W_TRICKY, "abcd", true},
    {W_TRICKY, "abcde", true},       {W_TRICKY, "abcdf", false},
    {W_TRICKY, "abcabcd", true},     {W_TRICKY, "d", true},
    {W_TRICKY, "e", true},           {W_EPSILON, "", true},
    {W_EPSILON, "a", false},         {W_BENCH2, "aaaa", true},
    {W_BENCH2, "abba", true},        {W_BENCH2, "aba", false},
    {W_BENCH2, "babbab", true},      {W_BENCH2, "bbbb", false},
};

// FNV-1a over every node's `marked`, pre-order — node, left subtree, right
// subtree.  The order is part of the digest, so two ports must also agree on
// the tree's shape and not only on its marks.
static uint64_t marks_digest(Node *root) {
    uint64_t h = 0xcbf29ce484222325ULL;
    std::vector<Node *> stack;
    stack.push_back(root);
    while (!stack.empty()) {
        Node *cur = stack.back();
        stack.pop_back();
        h = (h ^ (uint64_t)cur->marked) * 0x100000001b3ULL;
        if (cur->right) stack.push_back(cur->right);
        if (cur->left) stack.push_back(cur->left);
    }
    return h;
}

// `matches` without the `reset`, so the marks survive to be digested.
static int64_t scan_no_reset(Node *root, const uint8_t *s, size_t len) {
    if (len == 0) {
        return (int64_t)root->empty;
    }
    int64_t result = shift(root, (int64_t)s[0], 1);
    for (size_t i = 1; i < len; i++) {
        result = shift(root, (int64_t)s[i], 0);
    }
    return result;
}

// Two `a`s exactly `n + 1` apart is the whole regex.
static std::vector<uint8_t> plant(const std::vector<uint8_t> &s, size_t i, size_t n) {
    std::vector<uint8_t> out = s;
    out[i] = 'a';
    out[i + n + 1] = 'a';
    return out;
}

static int verify(size_t len, size_t n) {
    size_t total = sizeof(VECTORS) / sizeof(VECTORS[0]);
    size_t bad = 0;
    uint64_t answers = 0;
    int bit = 0;
    for (size_t i = 0; i < total; i++) {
        Node *root = build(VECTORS[i].w);
        const char *s = VECTORS[i].s;
        size_t slen = 0;
        while (s[slen]) slen++;
        bool got = matches(root, (const uint8_t *)s, slen);
        if (got) answers |= (uint64_t)1 << bit;
        bit++;
        if (got != VECTORS[i].want) {
            bad++;
            std::fprintf(stderr, "verify FAIL vector %zu: input \"%s\" got %d want %d\n", i, s,
                         (int)got, (int)VECTORS[i].want);
        }
    }

    std::vector<uint8_t> s = nonmatching(len, n, 42);
    Node *root = bench_regex(n);
    std::vector<std::vector<uint8_t>> cands = {s, plant(s, 0, n), plant(s, len / 2, n),
                                               plant(s, len - 1 - (n + 1), n)};
    for (size_t i = 0; i < cands.size(); i++) {
        if (matches(root, cands[i].data(), cands[i].size())) answers |= (uint64_t)1 << bit;
        bit++;
    }

    scan_no_reset(root, s.data(), s.size());
    uint64_t marks = marks_digest(root);
    reset(root);

    char head[65], tail[65];
    std::memcpy(head, s.data(), 64);
    head[64] = 0;
    std::memcpy(tail, s.data() + s.size() - 64, 64);
    tail[64] = 0;
    std::fprintf(stderr, "verify vectors %zu/%zu\n", total - bad, total);
    std::printf("verify nodes=%zu input_fnv1a=%016llx head=%s tail=%s answers=%016llx "
                "marks=%016llx\n",
                count_nodes(root), (unsigned long long)fnv1a64(s), head, tail,
                (unsigned long long)answers, (unsigned long long)marks);
    return bad == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--verify") == 0) {
        return verify((size_t)std::strtoull(argv[2], nullptr, 10),
                      argc > 3 ? (size_t)std::strtoull(argv[3], nullptr, 10) : 20);
    }
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <length> <repeats> [n]\n       %s --verify <length> [n]\n",
                     argv[0], argv[0]);
        return 2;
    }
    size_t len = (size_t)std::strtoull(argv[1], nullptr, 10);
    size_t repeats = (size_t)std::strtoull(argv[2], nullptr, 10);
    size_t n = argc > 3 ? (size_t)std::strtoull(argv[3], nullptr, 10) : 20;
    if (len == 0 || repeats == 0) {
        std::fprintf(stderr, "length and repeats must both be positive\n");
        return 2;
    }

    Node *root = bench_regex(n);
    size_t nodes = count_nodes(root);
    std::vector<uint8_t> s = nonmatching(len, n, 42);

    if (matches(root, s.data(), s.size())) {
        std::fprintf(stderr,
                     "the benchmark input matched: it is supposed NOT to, and a "
                     "matching input lets the scan stop early\n");
        return 1;
    }

    // One untimed round.  There is no JIT here to warm up, but the input is
    // 1 MiB and the tree wants to be in cache; timing the first touch of both
    // would report the page faults, not the matcher.
    (void)matches(root, s.data(), s.size());

    std::vector<double> rates;
    int64_t sink = 0;
    for (size_t r = 0; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        bool hit = matches(root, s.data(), s.size());
        auto t1 = std::chrono::steady_clock::now();
        sink += hit ? 1 : 0;
        double secs = std::chrono::duration<double>(t1 - t0).count();
        double rate = (double)len / secs;
        rates.push_back(rate);
        std::fprintf(stderr, "round %zu %.0f\n", r + 1, rate);
    }
    if (sink != 0) {
        std::fprintf(stderr, "a timed round reported a match; the number is not valid\n");
        return 1;
    }

    std::vector<double> sorted = rates;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];

    // The plausibility check the reader needs: chars/s alone cannot say whether
    // the tree walk survived compilation, but node visits per second can be
    // held against the machine's clock.
    std::fprintf(stderr, "detail nodes=%zu node_visits_per_s=%.0f sink=%lld input_fnv1a=%016llx\n",
                 nodes, median * (double)nodes, (long long)sink,
                 (unsigned long long)fnv1a64(s));
    std::printf("cpp %.0f\n", median);
    return 0;
}
