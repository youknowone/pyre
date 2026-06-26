class Obj:
    hits = 0

    @property
    def p(self):
        # A side-effecting getter: reading `obj.p` runs THIS Python frame,
        # whose body mutates `O.hits` (a class-attribute STORE_ATTR).  The
        # mutation is a concrete, NON-journaled, NON-idempotent heap write
        # that COMMITS before a later op in the same iteration aborts.
        Obj.hits += 1
        return 1


def f():
    obj = Obj()
    seen = []
    for x in range(500):
        # `obj.p` lowers to a value-returning `load_attr` residual
        # (`PyreHelperKind::None`, `Ref` result, `MayForce`): the OLD R1
        # write-discriminator (Void result OR a CallFn/StoreSubscr/
        # SetCurrentException/StoreDeref tag) does NOT see it, so its
        # getter side effect (`O.hits += 1`) escaped the body-effect flag.
        # `seen.append` is the abort trigger (its inline sub-walk declines)
        # AFTER the getter has already run its Python frame and committed
        # `O.hits += 1`.  Delivering the in-flight item and re-running the
        # body then runs the getter a SECOND time → `Obj.hits` DOUBLES
        # (500 + #aborts) on a re-run.
        #
        # The user-frame-entry signal flags the body effect: running the
        # getter entered a user Python frame after the in-flight FOR_ITER
        # consume, so `fbw_foriter_inflight_take` refuses delivery and the
        # legacy drop-on-abort fallback leaves `Obj.hits` EXACT at 500
        # (the bypass still advances the loop and runs the getter once per
        # iteration, including the aborted ones).
        v = obj.p
        seen.append(x)
    return Obj.hits


print(f())
