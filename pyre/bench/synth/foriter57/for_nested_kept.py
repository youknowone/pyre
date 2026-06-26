def f():
    acc = 0
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    # A nested loop whose INNER `for x in data` is the walk-entry / loop-header
    # FOR_ITER that the FBW walk traces and delivers an in-flight item from on a
    # trace abort.  `base` is an operand-stack-adjacent kept temp live across the
    # inner FOR_ITER, so the delivery's push lands only if the live frame is
    # PROVABLY at the inner loop's header state.
    #
    # `deliver_inflight_foriter_item` pushes the consumed item + repositions the
    # frame at `body_pc` (the FOR_ITER fallthrough), assuming the live frame is
    # parked at the loop-header FOR_ITER with the iterator on TOS.  `body_pc` is
    # nested-aware (derived from the consumed FOR_ITER op's OWN pc), so it can
    # name an inner FOR_ITER reached deeper in a traced body — where the live
    # frame is parked at the OUTER header, NOT the inner one.  Pushing there
    # corrupts the operand stack (a later GET_ITER/FOR_ITER reads a wrong slot
    # as an iterator -> "TypeError: not an iterator").  The header-state guard
    # delivers ONLY when the frame is parked at the FOR_ITER whose fallthrough
    # is `body_pc` (`next_instr() == body_pc - 1`, opcode there is FOR_ITER);
    # otherwise it REFUSES (drops the stash, the conservative drop-on-abort
    # fallback) rather than push to a non-header frame.  This case's inner
    # consume is header-consistent, so it STILL delivers and stays exact.
    for i in range(4000):
        base = i * 2
        for x in data:
            acc = acc + x + base
    return acc


print(f())
