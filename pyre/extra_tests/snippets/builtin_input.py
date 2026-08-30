import sys


class InputStream:
    def fileno(self):
        return 0

    def isatty(self):
        return True


class OutputStream:
    def __init__(self):
        self.text = []
        self.flushed = False

    def fileno(self):
        return 1

    def write(self, text):
        self.text.append(text)

    def flush(self):
        self.flushed = True


class ErrorStream:
    def __init__(self):
        self.flushed = False

    def flush(self):
        self.flushed = True


stdin, stdout, stderr = sys.stdin, sys.stdout, sys.stderr
missing = object()
raw_input = getattr(sys, "__raw_input__", missing)
try:
    fake_stdout = OutputStream()
    fake_stderr = ErrorStream()
    prompts = []

    def read_from_hook(prompt):
        prompts.append(prompt)
        return "answer"

    sys.stdin = InputStream()
    sys.stdout = fake_stdout
    sys.stderr = fake_stderr
    sys.__raw_input__ = read_from_hook

    assert input("prompt") == "answer"
    assert prompts == ["prompt"]
    assert "".join(fake_stdout.text) == ""
    assert fake_stdout.flushed
    assert fake_stderr.flushed

    class BadFilenoInput(InputStream):
        def fileno(self):
            raise RuntimeError("bad fileno")

        def readline(self):
            return "fallback\n"

    fake_stdout.text.clear()
    prompts.clear()
    sys.stdin = BadFilenoInput()
    assert input("ordinary") == "fallback"
    assert prompts == []
    assert "".join(fake_stdout.text) == "ordinary"
finally:
    sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr
    if raw_input is missing:
        del sys.__raw_input__
    else:
        sys.__raw_input__ = raw_input
