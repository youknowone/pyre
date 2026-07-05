# The walk loop's hot body reads `head.val` (an int) and `head.next` (the next
# node).  The full-body-walker LOAD_ATTR fold folds the object-typed `head.next`
# read to `guard_class` + `guard_value(map)` + `getfield(storage)` +
# `getarrayitem` (no residual — storage is a fixed-layout GcArray block); the
# `head.val` read stays a residual because `val` is an unboxed-int slot, whose
# read boxes a longlong rather than a plain fetch.  The point is to prove the
# `is not None` branch compiles (POP_JUMP_IF_NONE lowering) and the object-attr
# read folds inline, not to race pypy on the still-residual unboxed read.
#
# N/ITERS are kept small because the wasm backend runs every guard-exit
# re-entry through the not-yet-collected interpreter allocation path, so the
# residual `head.val` read leaks per re-entry and the wall grows super-linearly
# in ITERS on wasm (native dynasm/cranelift stay linear via bridge chaining).
# This is the pre-existing wasm interpreter-alloc leak, not the LOAD_ATTR fold;
# the size here leaves the wasm run well inside the synthetic timeout while
# still driving the compiled walk loop thousands of times on every backend.
N = 300
ITERS = 500


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
