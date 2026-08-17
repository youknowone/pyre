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

# A line of nothing but spaces or tabs holds nothing, so it is emptied rather
# than having the prefix taken off it, and its own depth never narrows the
# prefix the lines around it share.
done = subprocess.run(
    [sys.executable, "-c", '    s = """A\n        \n    B"""\n    print(repr(s))'],
    capture_output=True,
    text=True,
)
assert done.returncode == 0, (done.returncode, done.stderr)
assert done.stdout.strip() == repr("A\n\nB"), done.stdout

# Emptying those lines is part of removing a prefix, not something done on its
# own: an argument with a line at column zero has no prefix to remove and comes
# back exactly as it was written, blank lines included.
done = subprocess.run(
    [sys.executable, "-c", 'x = """a\n  \nb"""\nprint(repr(x))'],
    capture_output=True,
    text=True,
)
assert done.returncode == 0, (done.returncode, done.stderr)
assert done.stdout.strip() == repr("a\n  \nb"), done.stdout

# Only spaces and tabs make a line empty.  Every other whitespace character is
# content, so a line holding one has an indent of its own: at column zero it
# leaves no common prefix at all, and after two spaces it narrows the prefix to
# those two.  A carriage return is such a character, which is what makes a
# line ending in one something other than a blank line.
#
# The vertical tab and the information separators are left out: they are
# non-printable, and which of the two errors a tokenizer reports for one is not
# what this is testing.
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
