# CPython-suite gap: `test_frame` deletes through the proxy only to check that
# a fast local refuses, so nothing in the suite pins what `del` and `pop`
# report for a key that names no slot, or for a key that cannot be hashed.
#
# `framelocalsproxy_setitem` with no value, and `framelocalsproxy_pop`, both
# resolve the key through the same scan the subscript uses, and that scan
# hashes the key before it looks at any name.  Probing a materialized snapshot
# first instead gets the refusal right and everything else wrong: the miss
# arrives as the snapshot's `KeyError`, an unhashable key arrives in the dict's
# terms, and for `pop` the discarded probe turns the hash's `TypeError` into a
# `KeyError` naming the unhashable key.
#
# parity-tests reason: every line below is exception text, which is only worth
# anything next to the runtime it has to agree with.
import sys


def report(label, call):
    try:
        print(label, "->", call())
    except Exception as exc:
        print(label, "->", type(exc).__name__, exc.args)


def delete_and_pop():
    bound = 1  # noqa: F841 - named through the proxy below
    proxy = sys._getframe(0).f_locals
    # A key the frame has a slot for is refused whichever way it is asked.
    report("del local", lambda: proxy.__delitem__("bound"))
    report("pop local", lambda: proxy.pop("bound"))
    # The hash runs before the scan, so neither of these reaches the extras
    # dict to be described in its terms.
    report("del unhashable", lambda: proxy.__delitem__(["unhashable"]))
    report("pop unhashable", lambda: proxy.pop(["unhashable"]))
    # A frame with no extras dict yet reports the key itself, or the default.
    report("del absent", lambda: proxy.__delitem__("absent"))
    report("pop absent", lambda: proxy.pop("absent"))
    report("pop absent default", lambda: proxy.pop("absent", "fallback"))
    # Once the extras dict exists it answers both, and a second delete of the
    # same name is a miss again.
    proxy["extra"] = 7
    report("pop extra", lambda: proxy.pop("extra"))
    proxy["extra"] = 8
    report("del extra", lambda: proxy.__delitem__("extra"))
    report("del extra again", lambda: proxy.__delitem__("extra"))
    print("bound untouched", bound)


delete_and_pop()
print("OK")
