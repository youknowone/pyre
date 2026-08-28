class S(str):
    def __getitem__(self, index):
        return "override"


def item(value):
    return value[0]


# Record the exact-str specialization, then make a subclass flow through its
# guard.  The subclass must side-exit to its Python override.
for _ in range(500):
    assert item("abc") == "a"
assert item(S("abc")) == "override"

# Recording a subclass first must decline the exact-str specialization rather
# than pinning a canonical class that the recorded operand does not carry.
subclass = S("xyz")
for _ in range(500):
    assert item(subclass) == "override"
