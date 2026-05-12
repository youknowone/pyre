"""Nested try/except picks the innermost (last-matching-wins) handler.

`pypy/interpreter/pycode.py:250-253 lookup_exceptiontable` keeps walking
the table even after a match, choosing the *later* (innermost) entry
when multiple ranges cover the raise site.  Earlier pyre `.find(...)`
returned the first match, which diverges in nested cases.
"""


def go():
    log = []
    try:
        try:
            raise ValueError("inner")
        except KeyError:
            log.append("wrong-outer")
        except ValueError as e:
            log.append(f"inner-caught:{e}")
    except Exception as e:
        log.append(f"outer-caught:{e}")
    return log


def deeper():
    log = []
    try:
        try:
            try:
                raise RuntimeError("deep")
            except TypeError:
                log.append("wrong-1")
        except KeyError:
            log.append("wrong-2")
        except RuntimeError as e:
            log.append(f"deep-caught:{e}")
    except Exception as e:
        log.append(f"propagated:{e}")
    return log


assert go() == ["inner-caught:inner"], go()
assert deeper() == ["deep-caught:deep"], deeper()
print("OK")
