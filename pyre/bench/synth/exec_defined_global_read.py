N = 200


def main():
    # A function defined inside `exec(src, fresh_dict)` captures the fresh
    # dict as its `__globals__`.  Calling it enough to JIT-compile a trace
    # must keep resolving its module globals (LOAD_GLOBAL `ADDEND`) through
    # that dict object; no separate backing storage is involved.
    ns = {}
    exec(
        "ADDEND = 3\n"
        "def hot(n):\n"
        "    s = 0\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        s = s + i + ADDEND\n"
        "        i = i + 1\n"
        "    return s\n",
        ns,
    )
    hot = ns["hot"]
    total = 0
    for _ in range(N):
        total = total + hot(N)
    print(total)

    # STORE_GLOBAL inside an exec-defined function writes the exec dict.
    exec("def setter():\n global Z\n Z = 99\n", ns)
    ns["setter"]()
    print(ns.get("Z"))

    # A global mutated after exec is visible to the exec-defined reader.
    ns["ADDEND"] = 10
    print(hot(5))


main()
