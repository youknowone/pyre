# pyre-check: gate=1
rejected = False
try:
    class C(object, None):
        pass
except TypeError:
    rejected = True

assert rejected
