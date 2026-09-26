# Virtual int bounds on str.find / rfind / count must reach the search as
# machine ints. A pointer cast of the boxed bound crashes the residual.

N = 20000


def parse(s):
    s = ";" + str(s)
    plist = []
    start = 0
    while s.find(";", start) == start:
        start += 1
        end = s.find(";", start)
        ind, diff = start, 0
        while end > 0:
            diff += s.count('"', ind, end) - s.count('\\"', ind, end)
            if diff % 2 == 0:
                break
            end, ind = ind, s.find(";", end + 1)
        if end < 0:
            end = len(s)
        i = s.find("=", start, end)
        if i == -1:
            f = s[start:end]
        else:
            f = s[start:i].rstrip().lower() + "=" + s[i + 1 : end].lstrip()
        plist.append(f.strip())
        start = end
    return plist


def main():
    for s in ["", "foo=bar", " FOO = bar    "]:
        src = f"{s};" * (N - 1) + s
        res = parse(src)
        print(len(res), len(set(res)))


main()
