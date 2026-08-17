# pyre-check: gate=1
# The implicit `__class__` cellvar is never reassigned in the body,
# so MAKE_CELL must leave the pre-installed cell alone; a
# cell-of-cell would make zero-arg super() resolve an inner cell
# instead of the class.
class A:
    def f(self):
        return 1
class B(A):
    def f(self):
        return 10 + super().f()
result = B().f()

assert result == 11
