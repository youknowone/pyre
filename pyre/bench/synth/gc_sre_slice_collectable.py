# pyre-check: no-cpython

import gc
import re


def is_managed(value):
    return any(obj is value for obj in gc.get_objects())


pattern = re.compile(r"(?P<word>[a-z]+)-(?P<number>\d+)")
subject = "alpha-123-omega"
match = pattern.search(subject)

group_word = match.group(1)
getitem_number = match[2]
multiple = match.group(1, 2)
groups = match.groups()
groupdict = match.groupdict()
findall_plain = re.findall(r"[a-z]+", subject)
findall_single = re.findall(r"([a-z]+)", subject)
findall_multiple = pattern.findall(subject)
split_parts = re.split(r"(-)", subject)

assert group_word == "alpha"
assert getitem_number == "123"
assert multiple == ("alpha", "123")
assert groups == ("alpha", "123")
assert groupdict == {"word": "alpha", "number": "123"}
assert findall_plain == ["alpha", "omega"]
assert findall_single == ["alpha", "omega"]
assert findall_multiple == [("alpha", "123")]
assert split_parts == ["alpha", "-", "123", "-", "omega"]

assert is_managed(group_word)
assert is_managed(getitem_number)
assert is_managed(multiple[0])
assert is_managed(groups[1])
assert is_managed(groupdict["word"])
assert is_managed(findall_plain[0])
assert is_managed(findall_single[1])
assert is_managed(findall_multiple[0][1])
assert is_managed(split_parts[0])
assert is_managed(split_parts[1])


class Text(str):
    pass


unchanged = pattern.sub("replacement", Text("no match here"))
assert type(unchanged) is str
assert unchanged == "no match here"
assert is_managed(unchanged)

print("sre subject slices are collectable")
