def f():
    acc = 0
    # A nested FOR_ITER: the inner FOR_ITER is NOT the walk-entry / loop-header
    # opcode, so its body pc cannot be derived from the walk-entry coordinate.
    # The body_pc must come from the inner FOR_ITER op's OWN pc; a wrong body
    # pc would deliver the consumed item to the wrong opcode → silent
    # corruption or SIGBUS.
    for i in range(200):
        for j in range(50):
            acc += i * j
    return acc
print(f())
