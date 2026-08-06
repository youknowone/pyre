import io
import sys


N = 20000


def check_explicit_call():
    seen = set()
    for _ in range(N):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            seen.add(exc.__suppress_context__)
    assert seen == {False}, seen


def check_bare_class():
    seen = set()
    for _ in range(N):
        try:
            raise ValueError
        except ValueError as exc:
            seen.add(exc.__suppress_context__)
    assert seen == {False}, seen


def check_floor_divide():
    seen = set()
    zero = 0
    for value in range(N):
        try:
            value // zero
        except ZeroDivisionError as exc:
            seen.add(exc.__suppress_context__)
    assert seen == {False}, seen


def check_remainder():
    seen = set()
    zero = 0
    for value in range(N):
        try:
            value % zero
        except ZeroDivisionError as exc:
            seen.add(exc.__suppress_context__)
    assert seen == {False}, seen


def check_explicit_cause():
    seen = set()
    causes = set()
    cause = KeyError("cause")
    for _ in range(N):
        try:
            raise ValueError("outer") from cause
        except ValueError as exc:
            seen.add(exc.__suppress_context__)
            causes.add(exc.__cause__ is cause)
    assert seen == {True}, seen
    assert causes == {True}, causes


def check_rendered_implicit_context():
    rendered = ""
    for _ in range(N):
        try:
            try:
                raise KeyError("inner")
            except KeyError:
                raise ValueError("outer")
        except ValueError as exc:
            stream = io.StringIO()
            previous = sys.stderr
            sys.stderr = stream
            try:
                sys.excepthook(type(exc), exc, exc.__traceback__)
            finally:
                sys.stderr = previous
            rendered = stream.getvalue()
    assert "During handling of the above exception" in rendered, rendered


check_explicit_call()
check_bare_class()
check_floor_divide()
check_remainder()
check_explicit_cause()
check_rendered_implicit_context()
print("OK")
