def loop_with_two_backedges(n):
    high = 0
    for i in range(n):
        value = i % 7
        high = value if value > high else high
        if i % 45 == 0:
            high = 0
    return high


result = loop_with_two_backedges(4000)
assert result == 6
print(result)
