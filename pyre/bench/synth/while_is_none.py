# The walk loop's hot body is a single `head.next` LOAD_ATTR, which still
# emits a per-iteration `bh_load_attr_fn` residual (no inline-cache / map
# specialization yet), so this bench walks a slower attribute path than the
# arithmetic synth benches.  ITERS is sized so the compiled loop finishes well
# inside the synthetic timeout on every backend (cranelift is the slowest);
# the point is to prove the `is not None` branch compiles at all, not to race
# pypy on attribute loads.
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
