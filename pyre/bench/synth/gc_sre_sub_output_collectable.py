# pyre-check: no-cpython

import gc
import re


pattern = re.compile(r"([a-z]+)-(\d+)")
subject = "alpha-123-omega"

sub_result = pattern.sub("word", subject)
subn_result, count = pattern.subn("word", subject)
expand_result = pattern.search(subject).expand(r"<\1:\2>")

assert sub_result == "word-omega"
assert subn_result == "word-omega"
assert count == 1
assert expand_result == "<alpha:123>"
assert any(obj is sub_result for obj in gc.get_objects())
assert any(obj is subn_result for obj in gc.get_objects())
assert any(obj is expand_result for obj in gc.get_objects())

print("sre substitution outputs are collectable")
