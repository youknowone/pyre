import subprocess
import sys

# `-c` strips common leading whitespace before compiling an indented block.
indented = '\n    import sys\n    print("dedented")\n'
done = subprocess.run(
    [sys.executable, "-c", indented], capture_output=True, text=True
)
assert done.returncode == 0, (done.returncode, done.stderr)
assert done.stdout.strip() == "dedented", done.stdout

# A single line carries its own prefix, with no leading newline to start from.
done = subprocess.run(
    [sys.executable, "-c", '    print("one line")'], capture_output=True, text=True
)
assert done.returncode == 0, (done.returncode, done.stderr)
assert done.stdout.strip() == "one line", done.stdout

# Dedenting uses the raw text, including text inside string literals; a
# nonblank line at column zero therefore prevents all stripping.
done = subprocess.run(
    [sys.executable, "-c", '    x = 1\ny = 2\n'], capture_output=True, text=True
)
assert done.returncode != 0, done.stdout
assert "IndentationError" in done.stderr, done.stderr

# A tab and a space are different characters and never cancel, so a block
# indented with tabs is stripped by its tab.
done = subprocess.run(
    [sys.executable, "-c", '\n\tprint("tabbed")\n'], capture_output=True, text=True
)
assert done.returncode == 0, (done.returncode, done.stderr)
assert done.stdout.strip() == "tabbed", done.stdout

# Space/tab-only lines are emptied and do not narrow the common prefix.
done = subprocess.run(
    [sys.executable, "-c", '    s = """A\n        \n    B"""\n    print(repr(s))'],
    capture_output=True,
    text=True,
)
assert done.returncode == 0, (done.returncode, done.stderr)
assert done.stdout.strip() == repr("A\n\nB"), done.stdout

# Without a common prefix, the source is unchanged, including blank lines.
done = subprocess.run(
    [sys.executable, "-c", 'x = """a\n  \nb"""\nprint(repr(x))'],
    capture_output=True,
    text=True,
)
assert done.returncode == 0, (done.returncode, done.stderr)
assert done.stdout.strip() == repr("a\n  \nb"), done.stdout

# Whitespace other than space or tab is content: at column zero it prevents
# dedenting, and after two spaces it narrows the common prefix to two spaces.
#
# Non-printable vertical tabs and information separators are omitted because
# their tokenizer error category is unrelated to dedenting.
for content in ("\x0c", "\r", "  \r", "\xa0"):
    done = subprocess.run(
        [sys.executable, "-c", '    a = 1\n' + content + '\n    print("x")'],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0, (content, done.stdout)
    assert "IndentationError" in done.stderr, (content, done.stderr)

# A script file and code on stdin keep their indentation, which is still an
# error — the stripping belongs to `-c` alone.
done = subprocess.run(
    [sys.executable, "-"], input='    print("stdin")\n', capture_output=True, text=True
)
assert done.returncode != 0, done.stdout
assert "IndentationError" in done.stderr, done.stderr
