# Regression guard: a frame-forcing user @property that forces FIRST (the
# escape flush commits mid-property) and mutates SECOND (the commit is then
# withdrawn because the callee entered a user frame). The withdrawal must
# RESTORE the pre-flush frame before the legacy replay runs: replaying on the
# flushed (mid-iteration) frame skipped the journal-rolled-back list append,
# losing elements (len(seen) < loops, JIT-only).
import sys


class Obj:
    hits = 0

    @property
    def peek(self):
        _ = sys._getframe(1).f_locals  # forces the traced caller -> flush commits
        Obj.hits += 1                  # user-frame effect AFTER the commit
        return 1


o = Obj()


def main():
    seen = []
    total = 0
    for i in range(20000):
        seen.append(i)      # journaled append BEFORE the forcing read
        total += o.peek
    print(total, len(seen), sum(seen) % 1000003)


main()
