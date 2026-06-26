class Obj:
    hits = 0

    @property
    def p(self):
        # A side-effecting getter that MUTATES then RAISES: reading `obj.p`
        # runs THIS Python frame, whose body first mutates `Obj.hits` (a
        # class-attribute STORE_ATTR — a concrete, NON-journaled,
        # NON-idempotent heap write that COMMITS) and THEN raises.  The raise
        # routes the value-returning `load_attr` residual onto
        # `execute_residual_call`'s Err arm.
        Obj.hits += 1
        raise ValueError


def f():
    obj = Obj()
    seen = []
    for x in range(500):
        # `obj.p` lowers to a value-returning `load_attr` residual
        # (`PyreHelperKind::None`, `Ref` result, `MayForce`).  Its getter
        # frame mutates `Obj.hits` and raises; the raise takes the residual's
        # Err arm.  The R1 body-effect marking used to run ONLY on the Ok arm,
        # so the Err arm left the in-flight FOR_ITER item's body-effect flag
        # UNSET.  `seen.append` is the abort trigger (its inline sub-walk
        # declines) AFTER the getter already committed `Obj.hits += 1` and the
        # local `except` swallowed the raise.  With the flag unset,
        # `fbw_foriter_inflight_take` delivered the in-flight item and re-ran
        # the body, running the getter a SECOND time → `Obj.hits` DOUBLES
        # (500 + #aborts) on a re-run.
        #
        # Marking the body effect on BOTH arms (the raising user frame still
        # bumped the eval-loop entry odometer) makes `take` refuse delivery, so
        # the legacy drop-on-abort fallback leaves `Obj.hits` EXACT at 500.
        try:
            v = obj.p
        except ValueError:
            pass
        seen.append(x)
    return Obj.hits


print(f())
