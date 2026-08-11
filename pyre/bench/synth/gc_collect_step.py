import gc


if hasattr(gc, "collect_step"):
    states = gc.GcCollectStepStats
    assert states.GC_STATES == (
        "SCANNING",
        "MARKING",
        "SWEEPING",
        "FINALIZING",
        "USERDEL",
    )

    gc.disable()
    transitions = []
    while True:
        step = gc.collect_step()
        transitions.append((step.oldstate, step.newstate))
        assert step.count == 1
        assert step.duration == -1.0
        if step.major_is_done:
            break

    assert transitions[0] == (states.STATE_SCANNING, states.STATE_MARKING)
    assert transitions[-2:] == [
        (states.STATE_FINALIZING, states.STATE_USERDEL),
        (states.STATE_USERDEL, states.STATE_SCANNING),
    ]
    assert not gc.isenabled()

    try:
        step.count = 2
    except AttributeError:
        pass
    else:
        raise AssertionError("GcCollectStepStats.count must be read-only")


print("OK")
