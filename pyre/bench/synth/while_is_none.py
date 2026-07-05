# The walk loop's hot body is a single `head.next` LOAD_ATTR.  The mapdict
# fast path folds the type/version_tag/map guards and drops the getattr MRO
# walk, but the storage read is still a per-iteration residual call (instance
# storage is a Rust `Vec`, not an inline-readable GcArray), so this bench walks
# a heavier attribute path than the arithmetic synth benches.  ITERS is sized
# so the compiled loop finishes well inside the synthetic timeout on every
# backend (cranelift is the slowest); the point is to prove the `is not None`
# branch compiles and takes the attr fast path, not to race pypy on the read.
N = 3000
ITERS = 2000


class Node:
    __slots__ = ("val", "next")

    def __init__(self, val, nxt):
        self.val = val
        self.next = nxt


def build(n):
    head = None
    for i in range(n):
        head = Node(i, head)
    return head


def walk(head):
    # `while head is not None` compiles to POP_JUMP_IF_NONE.  Before that
    # opcode was lowered, its abort_permanent marker declined the whole
    # linked-list walk loop, forcing the interpreter.
    total = 0
    while head is not None:
        total += head.val
        head = head.next
    return total


def main():
    head = build(N)
    total = 0
    for _ in range(ITERS):
        total += walk(head)
    print(total)


main()
