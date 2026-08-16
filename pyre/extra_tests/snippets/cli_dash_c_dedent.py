import subprocess
import sys

# `-c` strips the common leading whitespace from its argument before compiling,
# so a code string written as an indented block does not have to carry the
# `if 1:` guard that once made one parseable.
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

# The prefix is the common one over the lines that hold something, taken from
# the raw text with no awareness of what is inside a string literal.  A line
# that reaches column zero leaves the whole argument alone.
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

# A script file and code on stdin keep their indentation, which is still an
# error — the stripping belongs to `-c` alone.
done = subprocess.run(
    [sys.executable, "-"], input='    print("stdin")\n', capture_output=True, text=True
)
assert done.returncode != 0, done.stdout
assert "IndentationError" in done.stderr, done.stderr
