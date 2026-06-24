import sys
seen = []
i = 0
while i < 805:
    i += 1
    seen.append(i)
sys.stdout.write(repr(seen[:6]) + " ... len=" + str(len(seen)))
