import re


for i in range(10_000):
    re.compile(str(i) + "|x")

print("OK")
