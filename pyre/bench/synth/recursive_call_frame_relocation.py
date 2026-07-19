# Deep self-recursion whose caller frame is relocated by a minor collection
# while the recursive callee runs.  The bytecode CALL fast path drops the
# arguments, runs the callee, then pushes its result onto the caller's value
# stack.  The callee allocates enough to trigger a minor collection that moves
# the caller frame, so the raw pointer captured before the call goes stale; the
# result push and its valuestackdepth bump must land on the forwarded live
# frame.  When they hit the abandoned copy the live frame keeps a stack depth
# one slot short, the following BINARY_OP reads the range iterator instead of
# the recursion result ("unsupported operand type(s) for +: 'range_iterator'
# and 'int'"), and the dropped exception segfaults.  cat(5) sums to 5! per
# call; 1000 outer iterations warm the JIT and sustain the allocation pressure
# that forces the relocation.  No max-pypy-ratio: branchy recursion is the
# architectural JIT gap, so this is a correctness/crash guard only.
def cat(n):
    if n <= 1:
        return 1
    r = 0
    for k in range(n):
        r += cat(n - 1)
    return r


total = 0
for i in range(1000):
    total += cat(5)
print(total)
