N = 3000
ITERS = 20000


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
