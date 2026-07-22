import random
import threading

random.seed(1234)

# random.randint
assert random.randint(1, 11) == 8

# random.shuffle
left = list(range(10))
right = [2, 7, 3, 5, 8, 4, 6, 9, 0, 1]
random.shuffle(left)
assert left == right

# random.choice
assert random.choice(left) == 5

# random.choices
expected = ["red", "green", "red", "black", "black", "red"]
result = random.choices(["red", "black", "green"], [18, 18, 2], k=6)
assert expected == result

# random.sample
sampled = [0, 2, 1]
assert random.sample(list(range(3)), 3) == sampled

# random.randbytes
random.seed(1234)
zero_size_buf = random.randbytes(0)
assert type(zero_size_buf) is bytes
assert len(zero_size_buf) == 0
non_zero_buf = random.randbytes(4)
assert type(non_zero_buf) is bytes
assert list(non_zero_buf) == list(b"\xb9\x7fi\xf7")


# `_random.descr_new__` allocates the requested subtype, and the underlying
# interp-level TypeDef is process-global rather than thread-local.
class RandomSubclass(random.Random):
    def __init__(self, *, marker):
        self.marker = marker


subclass_instance = RandomSubclass(marker=15)
assert type(subclass_instance) is RandomSubclass
assert subclass_instance.marker == 15

thread_result = []


def make_randoms_in_thread():
    base_instance = random.Random(23)
    subclass_instance = RandomSubclass(marker=7)
    thread_result.append(
        (
            type(base_instance) is random.Random,
            type(subclass_instance) is RandomSubclass,
            subclass_instance.marker,
        )
    )


thread = threading.Thread(target=make_randoms_in_thread)
thread.start()
thread.join(timeout=5)
assert not thread.is_alive()
assert thread_result == [(True, True, 7)]

# TODO : random.random(), random.uniform(), random.triangular(),
#        random.betavariate, random.expovariate, random.gammavariate,
#        random.gauss, random.lognormvariate, random.normalvariate,
#        random.vonmisesvariate, random.paretovariate, random.weibullvariate
