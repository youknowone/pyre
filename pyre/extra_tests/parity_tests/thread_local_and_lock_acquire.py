"""`_thread._local` per-thread initialization and lock acquire arguments.

PyPy `os_local.py:47-64 create_new_dict`:

    def create_new_dict(self, ec):
        # create a new dict for this thread
        space = ec.space
        w_dict = space.newdict(instance=True)
        self.dicts[ec] = w_dict
        # call __init__
        try:
            w_type = space.type(self)
            w_init = space.getattr(w_type, space.newtext("__init__"))
            space.call_obj_args(w_init, self, self.initargs)
        except:
            # failed, forget w_dict and propagate the exception
            del self.dicts[ec]
            raise
        # ready
        self._register_in_ec(ec)
        return w_dict

and `os_local.py:78-88 descr_local__new__`:

    if __args__.arguments_w or __args__.keyword_names_w:
        w_parent_init, _ = space.lookup_in_type_where(w_subtype, '__init__')
        if w_parent_init is space.w_object:
            raise oefmt(space.w_TypeError,
                        "Initialization arguments are not supported")

Pinned contract:
  1. a subclass initializer runs again on the first access from each
     thread, with the arguments the object was constructed from —
     keywords included,
  2. an initializer that raises propagates out of the attribute access
     and leaves no dictionary behind,
  3. construction arguments are accepted according to the initializer
     the subtype inherits, not the requested type,
  4. per-thread state is not shared,
  5. `os_lock.py:23-40 parse_acquire_args` rejects a timeout on a
     non-blocking call, a negative timeout, and one past `TIMEOUT_MAX`.
"""

import _thread


def run_in_thread(fn):
    done = _thread.allocate_lock()
    done.acquire()
    box = []

    def wrapper():
        try:
            box.append(("ok", fn()))
        except BaseException as exc:
            box.append(("err", type(exc).__name__, str(exc)))
        finally:
            done.release()

    _thread.start_new_thread(wrapper, ())
    done.acquire()
    done.release()
    return box[0]


# 1. The initializer re-runs per thread with the original arguments.
class Local(_thread._local):
    def __init__(self, a, b=0):
        self.a = a
        self.b = b


obj = Local(1, b=2)
assert (obj.a, obj.b) == (1, 2)
assert run_in_thread(lambda: (obj.a, obj.b)) == ("ok", (1, 2))


# 2. A failing initializer propagates and discards the half-built dict.
class Failing(_thread._local):
    count = 0

    def __init__(self):
        type(self).count += 1
        if type(self).count > 1:
            raise ValueError("boom")


failing = Failing()
assert run_in_thread(lambda: failing.__dict__) == ("err", "ValueError", "boom")


# 3. Argument acceptance follows the inherited initializer.
class NoInit(_thread._local):
    pass


for factory in (_thread._local, NoInit):
    try:
        factory(1)
    except TypeError as exc:
        assert str(exc) == "Initialization arguments are not supported", exc
    else:
        raise AssertionError("%r accepted an argument" % (factory,))

Local(1)
_thread._local()


# 4. Per-thread state is not shared.
plain = _thread._local()
plain.x = "main"
assert run_in_thread(lambda: getattr(plain, "x", "<unset>")) == ("ok", "<unset>")
assert plain.x == "main"


# 5. parse_acquire_args.
lock = _thread.allocate_lock()
assert lock.acquire(False) is True
assert lock.acquire(False) is False
assert lock.acquire(True, 0.05) is False
lock.release()

rlock = _thread.RLock()
assert rlock.acquire() is True
assert rlock.acquire() is True
rlock.release()
rlock.release()

# The wording of these differs between implementations; only the type is part
# of the interface.
for call in (
    lambda: lock.acquire(False, 1.0),
    lambda: lock.acquire(True, -2.0),
):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError("bad acquire arguments accepted")

try:
    lock.acquire(True, _thread.TIMEOUT_MAX * 2.0)
except OverflowError:
    pass
else:
    raise AssertionError("oversized timeout accepted")

print("OK")
