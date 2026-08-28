"""The marked-regex matcher in RPython, as "A JIT for Regular Expression
Matching" writes it -- a class hierarchy with `_immutable_fields_` on
everything but the mark, and one JitDriver whose green is the regex.

This is the side of the comparison the majit example is measured against.
It is deliberately NOT the lowered `NodeRec` graph the Rust side walks: the
post writes classes and lets RPython's rtyper lower them, so that is what
this writes. Both sides then run the same algorithm, each spelled the way
its own toolchain expects, which is the only comparison that means anything.

Run it with `runner.py`, which drives it through `LLJitMixin` -- the real
metainterpreter and the real optimizer -- and prints the optimized loop.
"""

from rpython.rlib import jit


class Regex(object):
    _immutable_fields_ = ['empty']

    def __init__(self, empty):
        self.empty = empty
        # The one mutable field. Every structural field above it is
        # immutable, which is what lets the tracer fold the walk away.
        self.marked = False

    def reset(self):
        self.marked = False

    def shift(self, c, mark):
        marked = self._shift(c, mark)
        self.marked = marked
        return marked

    def _shift(self, c, mark):
        raise NotImplementedError("abstract")


class Char(Regex):
    _immutable_fields_ = ['c']

    def __init__(self, c):
        Regex.__init__(self, False)
        self.c = c

    def _shift(self, c, mark):
        return mark and c == self.c


class Epsilon(Regex):
    def __init__(self):
        Regex.__init__(self, True)

    def _shift(self, c, mark):
        return False


class Binary(Regex):
    _immutable_fields_ = ['left', 'right']

    def __init__(self, left, right, empty):
        Regex.__init__(self, empty)
        self.left = left
        self.right = right


class Alternative(Binary):
    def __init__(self, left, right):
        Binary.__init__(self, left, right, left.empty or right.empty)

    def _shift(self, c, mark):
        marked_left = self.left.shift(c, mark)
        marked_right = self.right.shift(c, mark)
        return marked_left or marked_right


class Repetition(Regex):
    _immutable_fields_ = ['re']

    def __init__(self, re):
        Regex.__init__(self, True)
        self.re = re

    def _shift(self, c, mark):
        return self.re.shift(c, mark or self.marked)


class Sequence(Binary):
    def __init__(self, left, right):
        Binary.__init__(self, left, right, left.empty and right.empty)

    def _shift(self, c, mark):
        # The left mark from the PREVIOUS character is what enters the right
        # side, so read it before `shift` overwrites it.
        old_marked_left = self.left.marked
        marked_left = self.left.shift(c, mark)
        marked_right = self.right.shift(
            c, old_marked_left or (mark and self.left.empty))
        return (marked_left and self.right.empty) or marked_right


def reset(re):
    re.reset()
    if isinstance(re, Binary):
        reset(re.left)
        reset(re.right)
    elif isinstance(re, Repetition):
        reset(re.re)


jitdriver = jit.JitDriver(
    greens=['re'],
    reds=['i', 'result', 's'],
)


def match(re, s):
    """The post's portal: the regex is the green, the position is a red."""
    if len(s) == 0:
        return re.empty
    result = re.shift(ord(s[0]), True)
    i = 1
    while i < len(s):
        jitdriver.jit_merge_point(re=re, i=i, result=result, s=s)
        result = re.shift(ord(s[i]), False)
        i += 1
    reset(re)
    return result
