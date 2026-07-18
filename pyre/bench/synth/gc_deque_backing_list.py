# collections.deque holds its backing list solely through the deque object.
# If the marker traces the deque with an empty offset set, that list is not
# forwarded and is swept/moved on a collection driven by a hot allocator loop
# while the deque is a live root — use-after-free / silent corruption on the
# next len/index/iterate.  Each case holds a deque live across rounds of heavy
# nursery allocation (forcing minor + major collections) and folds its contents
# into a checksum every round; a dropped backing list crashes or diverges the
# checksum from the interpreter oracle.
import collections

N = 6000
ALLOC = 200


def churn():
    s = 0
    for i in range(ALLOC):
        x = [i, i + 1, i + 2]
        s += x[0] + x[2]
    return s


def append_hold():
    d = collections.deque()
    for i in range(10):
        d.append(i)
    acc = 0
    for _ in range(N):
        churn()
        acc = (acc + sum(d)) % 1000003
    return acc


def grow_each_round():
    d = collections.deque(range(5))
    acc = 0
    for k in range(N):
        churn()
        d.append(k)
        d.popleft()
        acc = (acc + len(d) + d[0]) % 1000003
    return acc


def deque_of_objects():
    d = collections.deque(str(i) for i in range(8))
    acc = 0
    for _ in range(N):
        churn()
        acc = (acc + sum(int(s) for s in d)) % 1000003
    return acc


def nested_field():
    class Box:
        def __init__(self):
            self.q = collections.deque(range(7))

    b = Box()
    acc = 0
    for _ in range(N):
        churn()
        acc = (acc + sum(b.q)) % 1000003
    return acc


print("append_hold", append_hold())
print("grow_each_round", grow_each_round())
print("deque_of_objects", deque_of_objects())
print("nested_field", nested_field())
