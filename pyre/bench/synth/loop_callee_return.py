# Regression: a loop-bearing callee whose return value is consumed by a hot
# caller loop. The inline recursive-call-assembler path (opimpl_recursive_call_
# assembler) reaches the callee's loop back-edge, pops the inline frame, and
# emits a CALL_ASSEMBLER into the callee loop's compiled token. The callee loop
# is virtualizable (index_of_virtualizable == 0, [frame, ec] inputargs), so it
# reads its locals from the callee frame written back by
# gen_writeback_inline_frame_to_heap. Exercises both consume (total += ...) and
# discard (bare call) shapes; the result must stay correct on both backends.


def sum_to(n):
    s = 0
    i = 0
    while i < n:
        s += i
        i += 1
    return s


def driver(rounds):
    total = 0
    k = 0
    while k < rounds:
        total += sum_to(50)   # return value consumed by the hot loop
        sum_to(20)            # return value discarded
        k += 1
    return total


print(driver(50000))
