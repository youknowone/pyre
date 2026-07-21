class ReentrantKey:
    def __init__(self, value, state):
        self.value = value
        self.state = state

    def __hash__(self):
        return 9

    def __eq__(self, other):
        marker = self.state[1]
        if marker is not None:
            self.state[0].add(marker)
        return isinstance(other, ReentrantKey) and self.value == other.value


state = [None, 999]
s = {ReentrantKey(1, state)}
state[0] = s

s.add(ReentrantKey(2, state))
print("add", len(s), 999 in s)

state[1] = 888
print("contains", ReentrantKey(2, state) in s, len(s), 888 in s)

state[1] = 777
s.discard(ReentrantKey(1, state))
print("discard", len(s), 777 in s)
