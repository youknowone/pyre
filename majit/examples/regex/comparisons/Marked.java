// !! THIS FILE HAS NEVER BEEN COMPILED OR RUN. !!
//
// The machine it was written on has no JDK: `/usr/bin/java` and
// `/usr/bin/javac` exist, but they are the macOS stubs and
// `/usr/libexec/java_home` finds no runtime behind them.  A filesystem sweep
// for any bundled JVM found none.  So every claim this file makes about
// behaviour is unexecuted, and no number should be quoted from it until it has
// run once.  `run.sh` is built so that the first run proves it rather than
// trusting it: it runs `--verify` before timing, and the row is skipped unless
// that line matches the other ports character for character.
//
// One command fixes this:  brew install --cask temurin
//
// The marked-regex matcher in Java — the "Java (Baltasar Trancon y Widemann)"
// row of "A JIT for Regular Expression Matching" (2010), which reported
// 1,920,000 chars/s on a 2.26 GHz Core 2 Duo and was the fastest of the
// hand-written implementations there.
//
// Transcribed from the crate's `src/interp.rs`: the same five node kinds, the
// same field set, the same `shift` arms in the same order, `empty` computed
// once while the tree is built, `&` / `|` rather than `&&` / `||` so the body
// carries no short-circuit branch, the tree built BALANCED the way
// `regex.rs::build_balanced` builds it, and the same input generator.
//
// WARM-UP: WARMUP_ROUNDS (5) untimed rounds over the same input before the
// first timed one.  HotSpot decides what to compile from invocation and
// back-edge counters, so the first pass through `matches` runs interpreted,
// then the 1M-iteration loop is replaced on-stack, and only after `matches`
// and `shift` have been entered enough times does the whole nest get compiled
// with a settled profile.  Timing round 1 would report that transition.  Five
// rounds at 2^20 characters is on the order of a hundred million `shift`
// invocations, far past every tier threshold; the per-round rates are printed
// on stderr so a reader can confirm they have stopped climbing rather than
// take that on trust.  If rounds 1..5 of the timed set are still rising, raise
// this constant — the number was warm-up, not steady state.
//
// The two ways this row could lie are the same as the C++ one's.  It cannot
// exit early: `matches` is a loop over the whole input with no `break`, and the
// answer is checked to be "no match".  And the work cannot be optimized away
// unobserved: `sink` accumulates the answers and is printed, the mark stores
// land in heap objects the tree still references, and `run.sh` runs the
// external control — the per-character rate must scale as 1 / (node count),
// so `n = 2` (21 nodes) must come out roughly 4.4x faster than `n = 20`
// (93 nodes).
//
// `java Marked --verify <length> [n]` is the correctness gate: it prints a line
// that every port must print identically — the input digest, the answers to a
// fixed battery, and a digest of all 93 marks left after scanning the benchmark
// input.  See the `verify` section below for why each part is there.  It
// catches the third way this row could lie: a matcher that lost an arm and
// answers "no match" to everything would pass the check above and post a
// number.
//
// Usage: java Marked <length> <repeats> [n]     (n defaults to 20)
// Prints one line on stdout: `java <chars_per_second>` (median of the rounds).
// Per-round detail goes to stderr as `round <i> <chars_per_second>`.

import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public final class Marked {

    static final byte KIND_CHAR = 0;
    static final byte KIND_EPSILON = 1;
    static final byte KIND_ALTERNATIVE = 2;
    static final byte KIND_SEQUENCE = 3;
    static final byte KIND_REPETITION = 4;

    /** How many untimed rounds run before the first timed one. See the file header. */
    static final int WARMUP_ROUNDS = 5;

    // `kind`, `ch`, `empty`, `left` and `right` are fixed once the tree is
    // built; `marked` is the single mutable bit the algorithm shifts around.
    // The four tags are `byte` rather than `int` so a node stays close to the
    // 24-byte Rust `NodeRec` — a matcher whose working set is a different size
    // is measuring a different cache.
    static final class Node {
        byte kind;
        byte ch;
        byte empty;
        byte marked;
        Node left;
        Node right;

        Node(byte kind, byte ch, byte empty, Node left, Node right) {
            this.kind = kind;
            this.ch = ch;
            this.empty = empty;
            this.marked = 0;
            this.left = left;
            this.right = right;
        }
    }

    static Node nodeChar(char c) {
        return new Node(KIND_CHAR, (byte) c, (byte) 0, null, null);
    }

    static Node nodeEpsilon() {
        return new Node(KIND_EPSILON, (byte) 0, (byte) 1, null, null);
    }

    static Node nodeAlt(Node l, Node r) {
        return new Node(KIND_ALTERNATIVE, (byte) 0, (byte) (l.empty | r.empty), l, r);
    }

    static Node nodeSeq(Node l, Node r) {
        return new Node(KIND_SEQUENCE, (byte) 0, (byte) (l.empty & r.empty), l, r);
    }

    static Node nodeRep(Node inner) {
        return new Node(KIND_REPETITION, (byte) 0, (byte) 1, inner, null);
    }

    // `a|b`.  A fresh pair every call: marks live in the nodes, so an instance
    // may never be shared — the twenty `(a|b)` groups of `(a|b){20}` are twenty
    // distinct objects.
    static Node ab() {
        return nodeAlt(nodeChar('a'), nodeChar('b'));
    }

    // Balanced association, matching `regex.rs::build_balanced`: left half is
    // `xs[lo .. mid)`, right half is `xs[mid .. hi)`.
    static Node buildBalanced(List<Node> xs, int lo, int hi) {
        if (hi - lo == 1) {
            return xs.get(lo);
        }
        int mid = lo + (hi - lo) / 2;
        Node l = buildBalanced(xs, lo, mid);
        Node r = buildBalanced(xs, mid, hi);
        return nodeSeq(l, r);
    }

    // `(a|b)*a(a|b){n}a(a|b)*`, the benchmark regex of the post.
    static Node benchRegex(int n) {
        List<Node> parts = new ArrayList<>();
        parts.add(nodeRep(ab()));
        parts.add(nodeChar('a'));
        for (int i = 0; i < n; i++) {
            parts.add(ab());
        }
        parts.add(nodeChar('a'));
        parts.add(nodeRep(ab()));
        return buildBalanced(parts, 0, parts.size());
    }

    static int countNodes(Node n) {
        int c = 1;
        if (n.left != null) {
            c += countNodes(n.left);
        }
        if (n.right != null) {
            c += countNodes(n.right);
        }
        return c;
    }

    // Push `mark` into `n` for input character `c`, and answer the mark that
    // comes out on the right.  The mark left inside `n` is the one this call
    // computed.
    //
    // `&` and `|`, never `&&` and `||`: both sides must run, and the point of
    // the substitution is that the body has no conditional branch in it.
    static long shift(Node n, long c, long mark) {
        long m;
        switch (n.kind) {
            case KIND_CHAR:
                m = mark & ((long) n.ch == c ? 1L : 0L);
                break;
            case KIND_EPSILON:
                m = 0L;
                break;
            case KIND_ALTERNATIVE: {
                long ml = shift(n.left, c, mark);
                long mr = shift(n.right, c, mark);
                m = ml | mr;
                break;
            }
            case KIND_REPETITION:
                m = shift(n.left, c, mark | (long) n.marked);
                break;
            case KIND_SEQUENCE: {
                // The left mark from the PREVIOUS character is what enters the
                // right side, so read it before `shift` overwrites it.
                long oldMarkedLeft = n.left.marked;
                long markedLeft = shift(n.left, c, mark);
                long markedRight =
                        shift(n.right, c, oldMarkedLeft | (mark & (long) n.left.empty));
                m = (markedLeft & (long) n.right.empty) | markedRight;
                break;
            }
            default:
                m = 0L;
                break;
        }
        n.marked = (byte) m;
        return m;
    }

    // Clear every mark, so the graph can match another string.
    static void reset(Node n) {
        n.marked = 0;
        if (n.left != null) {
            reset(n.left);
        }
        if (n.right != null) {
            reset(n.right);
        }
    }

    // Shift one mark in from the left for `s[0]`, then shift the marks already
    // inside the graph along for every remaining character.  No `break`: every
    // character is looked at, whatever the answer turns out to be.
    static boolean matches(Node root, byte[] s) {
        if (s.length == 0) {
            return root.empty != 0;
        }
        long result = shift(root, s[0], 1L);
        for (int i = 1; i < s.length; i++) {
            result = shift(root, s[i], 0L);
        }
        reset(root);
        return result != 0;
    }

    // A random `a`/`b` string forced NOT to match `(a|b)*a(a|b){n}a(a|b)*`.
    //
    // Byte for byte the generator in `regex.rs::nonmatching`.  Java's `long` is
    // signed, but multiply and add wrap on the same 64 bits as Rust's
    // `wrapping_mul` / `wrapping_add`, so the states are identical; `>>>` is
    // what keeps the bit selection unsigned.
    static byte[] nonmatching(int len, int n, long seed) {
        byte[] s = new byte[len];
        for (int i = 0; i < len; i++) {
            seed = seed * 6364136223846793005L + 1442695040888963407L;
            s[i] = ((seed >>> 33) & 1L) == 0L ? (byte) 'a' : (byte) 'b';
        }
        int d = n + 1;
        for (int i = 0; i + d < len; i++) {
            if (s[i] == 'a' && s[i + d] == 'a') {
                s[i + d] = 'b';
            }
        }
        return s;
    }

    // A digest of the input, so `run.sh` can prove every port scanned the same
    // bytes.  Any hash would do; FNV-1a is four lines in all four languages.
    static long fnv1a64(byte[] data) {
        long h = 0xcbf29ce484222325L;
        for (byte b : data) {
            h = (h ^ (b & 0xFFL)) * 0x100000001b3L;
        }
        return h;
    }

    // ── verify ─────────────────────────────────────────────────────────────
    //
    // `--verify` is the correctness gate, and it is built so that the four
    // ports can be checked against EACH OTHER and against the crate's
    // `interp::matches`, not merely each against its own copy of an
    // expectation.  It prints one line, and every port must print the same one:
    //
    //   verify nodes=.. input_fnv1a=.. head=.. tail=.. answers=.. marks=..
    //
    //   * `input_fnv1a`, `head`, `tail` — the bytes this port generated.  A
    //     port whose LCG is off by one wrapping multiply still produces a
    //     plausible `a`/`b` string, and a chars/s number taken over different
    //     bytes is not a row in the same table.  Java's `long` is signed where
    //     the generator's is not, so this is the line that settles whether
    //     `>>>` and the wrapping multiply really did reproduce the u64
    //     arithmetic, rather than an argument that they must have.
    //   * `answers` — one bit per case of a fixed battery: the 28 vectors of
    //     `regex.rs::vectors()`, then four cases at the benchmark's own scale
    //     (the non-matching input, and the same input with a matching pair of
    //     `a`s planted at the front, in the middle, and hard against the end).
    //     The planted cases catch a matcher that lost an arm and answers "no"
    //     to everything — which a non-matching benchmark alone cannot catch —
    //     and the last one only matches if the final byte was read.
    //   * `marks` — a digest of all 93 `marked` bits left in the tree after
    //     scanning the benchmark input, taken BEFORE `reset` clears them.  This
    //     is the strong one: it compares the whole state of the computation
    //     after a million characters, so two ports agreeing on it are doing the
    //     same work and not merely arriving at the same boolean.
    //
    // The 28 vectors are also checked against their expected answers here, so a
    // port that agrees with the others but disagrees with `regex.rs` still
    // fails.

    static Node reAbc() { // `abc`
        return nodeSeq(nodeSeq(nodeChar('a'), nodeChar('b')), nodeChar('c'));
    }

    static Node reAbcAlt() { // `a|b|c`
        return nodeAlt(nodeAlt(nodeChar('a'), nodeChar('b')), nodeChar('c'));
    }

    static Node reTricky() { // `((abc)*|(abcd))(d|e)`
        Node l = nodeAlt(nodeRep(reAbc()), nodeSeq(reAbc(), nodeChar('d')));
        Node r = nodeAlt(nodeChar('d'), nodeChar('e'));
        return nodeSeq(l, r);
    }

    static final int W_ABC_ALT = 0;
    static final int W_REP_ABC_ALT = 1;
    static final int W_ABC = 2;
    static final int W_TRICKY = 3;
    static final int W_EPSILON = 4;
    static final int W_BENCH2 = 5;

    static Node build(int w) {
        switch (w) {
            case W_ABC_ALT: return reAbcAlt();
            case W_REP_ABC_ALT: return nodeRep(reAbcAlt());
            case W_ABC: return reAbc();
            case W_TRICKY: return reTricky();
            case W_EPSILON: return nodeEpsilon();
            default: return benchRegex(2);
        }
    }

    // `regex.rs::vectors()`, in its order.
    static final Object[][] VECTORS = {
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
    // subtree.  The order is part of the digest, so two ports must also agree
    // on the tree's shape and not only on its marks.
    static long marksDigest(Node root) {
        long h = 0xcbf29ce484222325L;
        ArrayDeque<Node> stack = new ArrayDeque<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            Node cur = stack.pop();
            h = (h ^ (cur.marked & 0xFFL)) * 0x100000001b3L;
            if (cur.right != null) {
                stack.push(cur.right);
            }
            if (cur.left != null) {
                stack.push(cur.left);
            }
        }
        return h;
    }

    // `matches` without the `reset`, so the marks survive to be digested.
    static long scanNoReset(Node root, byte[] s) {
        if (s.length == 0) {
            return root.empty;
        }
        long result = shift(root, s[0], 1L);
        for (int i = 1; i < s.length; i++) {
            result = shift(root, s[i], 0L);
        }
        return result;
    }

    // Two `a`s exactly `n + 1` apart is the whole regex.
    static byte[] plant(byte[] s, int i, int n) {
        byte[] out = s.clone();
        out[i] = 'a';
        out[i + n + 1] = 'a';
        return out;
    }

    static int verify(int len, int n) {
        int bad = 0;
        long answers = 0;
        int bit = 0;
        for (int i = 0; i < VECTORS.length; i++) {
            Node root = build((Integer) VECTORS[i][0]);
            String str = (String) VECTORS[i][1];
            boolean want = (Boolean) VECTORS[i][2];
            boolean got = matches(root, str.getBytes(StandardCharsets.US_ASCII));
            if (got) {
                answers |= 1L << bit;
            }
            bit++;
            if (got != want) {
                bad++;
                System.err.printf(Locale.ROOT, "verify FAIL vector %d: input \"%s\" got %b want %b%n",
                        i, str, got, want);
            }
        }

        byte[] s = nonmatching(len, n, 42L);
        Node root = benchRegex(n);
        byte[][] cands = {s, plant(s, 0, n), plant(s, len / 2, n), plant(s, len - 1 - (n + 1), n)};
        for (int i = 0; i < cands.length; i++) {
            if (matches(root, cands[i])) {
                answers |= 1L << bit;
            }
            bit++;
        }

        scanNoReset(root, s);
        long marks = marksDigest(root);
        reset(root);

        String head = new String(s, 0, 64, StandardCharsets.US_ASCII);
        String tail = new String(s, s.length - 64, 64, StandardCharsets.US_ASCII);
        System.err.printf(Locale.ROOT, "verify vectors %d/%d%n", VECTORS.length - bad,
                VECTORS.length);
        System.out.printf(Locale.ROOT,
                "verify nodes=%d input_fnv1a=%016x head=%s tail=%s answers=%016x marks=%016x%n",
                countNodes(root), fnv1a64(s), head, tail, answers, marks);
        return bad == 0 ? 0 : 1;
    }

    public static void main(String[] args) {
        if (args.length >= 2 && args[0].equals("--verify")) {
            System.exit(verify(Integer.parseInt(args[1]),
                    args.length > 2 ? Integer.parseInt(args[2]) : 20));
        }
        if (args.length < 2) {
            System.err.println("usage: java Marked <length> <repeats> [n]");
            System.err.println("       java Marked --verify <length> [n]");
            System.exit(2);
        }
        int len = Integer.parseInt(args[0]);
        int repeats = Integer.parseInt(args[1]);
        int n = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        if (len <= 0 || repeats <= 0) {
            System.err.println("length and repeats must both be positive");
            System.exit(2);
        }

        Node root = benchRegex(n);
        int nodes = countNodes(root);
        byte[] s = nonmatching(len, n, 42L);

        if (matches(root, s)) {
            System.err.println("the benchmark input matched: it is supposed NOT to, and a "
                    + "matching input lets the scan stop early");
            System.exit(1);
        }

        long warmSink = 0;
        for (int r = 0; r < WARMUP_ROUNDS; r++) {
            warmSink += matches(root, s) ? 1 : 0;
        }

        double[] rates = new double[repeats];
        long sink = warmSink;
        for (int r = 0; r < repeats; r++) {
            long t0 = System.nanoTime();
            boolean hit = matches(root, s);
            long t1 = System.nanoTime();
            sink += hit ? 1 : 0;
            double rate = (double) len / ((t1 - t0) / 1e9);
            rates[r] = rate;
            System.err.printf(Locale.ROOT, "round %d %.0f%n", r + 1, rate);
        }
        if (sink != 0) {
            System.err.println("a round reported a match; the number is not valid");
            System.exit(1);
        }

        double[] sorted = rates.clone();
        Arrays.sort(sorted);
        double median = sorted[sorted.length / 2];

        // The plausibility check: chars/s alone cannot say whether the tree
        // walk survived compilation, but node visits per second can be held
        // against the machine's clock.
        System.err.printf(Locale.ROOT,
                "detail nodes=%d warmup=%d node_visits_per_s=%.0f sink=%d input_fnv1a=%016x%n",
                nodes, WARMUP_ROUNDS, median * nodes, sink, fnv1a64(s));
        System.out.printf(Locale.ROOT, "java %.0f%n", median);
    }
}
