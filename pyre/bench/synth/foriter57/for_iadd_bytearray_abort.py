def f():
    # `acc += delta` for a `bytearray` mutates the receiver in place, but its
    # backing cannot be rewound by the Integer-strategy list journal, so the walk
    # DECLINES this loop (InplaceContainerMutationUnsupported) to the interpreter
    # rather than risk a re-delivery double or a dropped tail.  Same shape as
    # `for_iadd_list_abort` but a non-journalable container.
    #
    # Asserts BOTH the length (no double) AND the branch counter `n` (no dropped
    # tail); under the decline the whole loop runs interpreted, so it matches the
    # PYRE_NO_JIT oracle exactly.
    acc = bytearray()
    delta = bytearray(b"\x01")
    n = 0
    for x in range(500):
        acc += delta
        if x < 250:
            n += 1
    return len(acc), n


print(f())
