def inner():
    # 3.14 DUPLICATES the `finally` body into a normal-path copy and an
    # exceptional-path copy, so this loop appears as two FOR_ITER instructions
    # (the second inside an exception-table range).  The JIT compiles only the
    # normal copy; on loop exhaustion its side-exit resumes through the dense
    # carry-forward pc_map, which collapses the un-traced exhaustion-exit and the
    # exceptional copy into a single marker, so the resume lands at the
    # exceptional-copy FOR_ITER with an empty value stack ("stack underflow during
    # interpreter peek").  The resume coordinate was never emitted, so the frame
    # is declined to the interpreter (for_iter_frame_is_finally_duplicated).
    try:
        return "tryval"
    finally:
        acc = 0
        for x in range(2000):
            acc += x
        return ("FIN", acc)


for _ in range(300):
    r = inner()
print(r)
